use std::{borrow::Cow, collections::HashMap, hash::Hash, sync::{Arc, LazyLock, Mutex}};

use pyo3::{exceptions::{PyKeyError, PyValueError}, intern, prelude::*, types::{PyBytes, PyType}};
use tribles::{self, fucid, genid, query::{Binding, ConstantConstraint, Constraint, IntersectionConstraint, Query, TriblePattern, Variable}, valueschemas::UnknownValue, trible::{Trible, TRIBLE_LEN}, ufoid, RawId, TribleSet, Value};

use hex::FromHex;

struct PyPtrIdentity<T>(pub Py<T>);

impl<T> PartialEq for PyPtrIdentity<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

impl<T> Eq for PyPtrIdentity<T> {}

impl<T> Hash for PyPtrIdentity<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

static TYPE_TO_ENTITY: LazyLock<Mutex<HashMap<PyPtrIdentity<PyType>, RawId>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

static CONVERTERS: LazyLock<Mutex<HashMap<(RawId, RawId), Py<PyAny>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

#[pyfunction]
pub fn register_type(type_id: Py<PyId>, typ: Py<PyType>) {
    let mut type_to_entity = TYPE_TO_ENTITY.lock().unwrap();
    type_to_entity.insert(PyPtrIdentity(typ), type_id.get().bytes);
}

#[pyfunction]
pub fn register_converter(schema_id: Py<PyId>, typ: Py<PyType>, converter: Py<PyAny>) -> PyResult<()> {
    let type_id = {
        let type_to_entity = TYPE_TO_ENTITY.lock().unwrap();
        let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
            return Err(PyErr::new::<PyKeyError, _>("type should be registered first"));
        };
        entity.clone()
    };
    let mut converters = CONVERTERS.lock().unwrap();
    converters.insert((schema_id.get().bytes, type_id), converter);
    Ok(())
}

#[pyclass(frozen, name = "Id")]
pub struct PyId {
    bytes: [u8; 16],
}

#[pymethods]
impl PyId {
    #[new]
    fn new(bytes: &[u8]) -> Result<Self, PyErr> {
        let Ok(bytes) = bytes.try_into() else {
            return Err(PyValueError::new_err("ids should be 16 bytes"));
        };
        Ok(PyId {
            bytes,
        })
    }

    #[staticmethod]
    pub fn genid() -> Self {
        PyId { bytes: genid()}
    }

    #[staticmethod]
    pub fn ufoid() -> Self {
        PyId { bytes: ufoid()}
    }

    #[staticmethod]
    pub fn fucid() -> Self {
        PyId { bytes: fucid()}
    }

    #[staticmethod]
    pub fn hex(hex: &str) -> Result<Self, PyErr> {
        let Ok(bytes) = <[u8; 16]>::from_hex(hex) else {
            return Err(PyValueError::new_err("failed to parse hex id"));
        };
        Ok(PyId {
            bytes,
        })
    }

    pub fn to_hex(&self) -> String {
        hex::encode_upper(self.bytes)
    }

    pub fn bytes(&self) -> Cow<[u8]> {
        (&self.bytes).into()
    }
}

#[pyclass(frozen, name = "Value")]
pub struct PyValue {
    bytes: [u8; 32],
    _value_schema: [u8; 16],
    _blob_schema: [u8; 16]
}

#[pymethods]
impl PyValue {
    #[new]
    fn new(bytes: &[u8], value_schema: Py<PyId>, blob_schema: Option<Py<PyId>>) -> Self {
        let value_schema = value_schema.get().bytes;
        let blob_schema = blob_schema.map(|s| s.get().bytes).unwrap_or([0; 16]);
        PyValue {
            bytes: bytes.try_into().expect("values should be 32 bytes"),
            _value_schema: value_schema,
            _blob_schema: blob_schema,
        }
    }

    #[staticmethod]
    fn of(py: Python<'_>, schema: Py<PyId>, value: Bound<'_, PyAny>) -> PyResult<Self> {
        let value_schema = schema.get().bytes;
        let type_id = {
            let typ = value.get_type().unbind();
            let type_to_entity = TYPE_TO_ENTITY.lock().unwrap();
            let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
                return Err(PyErr::new::<PyKeyError, _>("type should be registered first"));
            };
            entity.clone()
        };
        let converters = CONVERTERS.lock().unwrap();
        let Some(converter) = converters.get(&(value_schema, type_id)) else {
            return Err(PyErr::new::<PyKeyError, _>("converter should be registered first"));
        };
        let bytes = converter.call_method_bound(py, intern!(py, "pack"), (value, ), None)?;
        let bytes = bytes.downcast_bound::<PyBytes>(py)?;
        let bytes: [u8; 32] = bytes.as_bytes().try_into()?;
        Ok(Self {
            bytes,
            _value_schema: value_schema,
            _blob_schema: [0; 16],
        })
    }

    fn to(&self, py: Python<'_>, typ: Py<PyType>) -> PyResult<Py<PyAny>> {
        let type_id = {
            let type_to_entity = TYPE_TO_ENTITY.lock().unwrap();
            let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
                return Err(PyErr::new::<PyKeyError, _>("type should be registered first"));
            };
            entity.clone()
        };
        let converters = CONVERTERS.lock().unwrap();
        let Some(converter) = converters.get(&(self._value_schema, type_id)) else {
            return Err(PyErr::new::<PyKeyError, _>("converter should be registered first"));
        };
        let bytes = PyBytes::new_bound(py, &self.bytes);
        converter.call_method_bound(py, intern!(py, "unpack"), (bytes,), None)
    }

    pub fn value_schema(&self) -> PyId {
        PyId {
            bytes: self._value_schema
        }
    }

    pub fn blob_schema(&self) -> Option<PyId> {
        if self._blob_schema == [0; 16] {
            None
        } else {
            Some(PyId {
                bytes: self._blob_schema
            })
        }
    }

    pub fn bytes(&self) -> Cow<[u8]> {
        (&self.bytes).into()
    }
}

#[pyclass(name = "TribleSet")]
pub struct PyTribleSet(tribles::TribleSet);

#[pymethods]
impl PyTribleSet {
    #[staticmethod]
    pub fn from_bytes(tribles: &Bound<'_, PyBytes>) -> Self {
        let tribles = tribles.as_bytes();
        assert!(tribles.len() % TRIBLE_LEN == 0);

        let mut set = tribles::TribleSet::new();

        for trible in tribles.chunks_exact(TRIBLE_LEN) {
            set.insert_raw(trible.try_into().unwrap());
        }

        PyTribleSet(set)
    }

    #[staticmethod]
    pub fn empty() -> Self {
        PyTribleSet(tribles::TribleSet::new())
    }

    pub fn __add__(&self, other: &Bound<'_, Self>) -> Self {
        let mut result = self.0.clone();
        result.union(other.borrow().0.clone());
        PyTribleSet(result)
    }

    pub fn __iadd__(&mut self, other: &Bound<'_, Self>) {
        let set = &mut self.0;
        set.union(other.borrow().0.clone());
    }

    pub fn __len__(&self) -> usize {
        return self.0.eav.len() as usize;
    }

    pub fn fork(&mut self) -> Self {
        PyTribleSet(self.0.clone())
    }

    pub fn add(&mut self, e: Py<PyId>,  a: Py<PyId>,  v: Py<PyValue>) {
        self.0.insert(&Trible::new(e.get().bytes, a.get().bytes, Value::<UnknownValue>::new(v.get().bytes)));
    }

    pub fn consume(&mut self, other: &Bound<'_, Self>) {
        let set = &mut self.0;
        let other_set = std::mem::replace(&mut other.borrow_mut().0, TribleSet::new());
        set.union(other_set);
    }

    pub fn pattern(&self, ev: u8, av: u8, vv: u8) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.pattern(Variable::new(ev), Variable::new(av), Variable::<UnknownValue>::new(vv)))
        }
    }
}

#[pyclass(name = "Query")]
pub struct PyQuery {
    query: Query<Arc<dyn Constraint<'static> + Send + Sync>, Box<dyn Fn(&Binding) -> Vec<PyValue> + Send>, Vec<PyValue>>
}

#[pyclass(frozen)]
pub struct PyConstraint {
    constraint: Arc<dyn Constraint<'static> + Send + Sync>
}

/// Build a constraint for the intersection of the provided constraints.
#[pyfunction]
pub fn constant(index: u8, constant: &Bound<'_, PyValue>) -> PyConstraint {
    let constraint = Arc::new(ConstantConstraint::new(
        Variable::<UnknownValue>::new(index),
        Value::<UnknownValue>::new(constant.get().bytes)));

    PyConstraint {
        constraint
    }
}


/// Build a constraint for the intersection of the provided constraints.
#[pyfunction]
pub fn intersect(constraints: Vec<Py<PyConstraint>>) -> PyConstraint {
    let constraints = constraints.iter().map(|py| py.get().constraint.clone()).collect();
    let constraint = Arc::new(IntersectionConstraint::new(constraints));

    PyConstraint {
        constraint
    }
}

/// Find solutions for the provided constraint.
#[pyfunction]
pub fn solve(projected: Vec<(u8, Py<PyId>)> ,constraint: &Bound<'_, PyConstraint>) -> PyQuery {
    let constraint = constraint.get().constraint.clone();

    let postprocessing = Box::new(move |binding: &Binding| {
        let mut vec = vec![];
        for (k, v) in &projected {
            vec.push(PyValue {
                bytes: *binding.get(*k).expect("constraint should contain projected variables"),
                _value_schema: v.get().bytes
            });
        }
        vec
    }) as Box<dyn Fn(&Binding) -> Vec<PyValue> + Send>;

    let query = tribles::query::Query::new(constraint, postprocessing);

    PyQuery {
        query
    }
}

#[pymethods]
impl PyQuery {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Vec<PyValue>> {
        slf.query.next()
    }
}

/// The `tribles` python module.
#[pymodule]
#[pyo3(name = "tribles")]
pub fn tribles_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    //m.add_class::<PyValue>()?;
    m.add_class::<PyTribleSet>()?;
    m.add_class::<PyId>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyQuery>()?;
    m.add_function(wrap_pyfunction!(register_type, m)?)?;
    m.add_function(wrap_pyfunction!(register_converter, m)?)?;
    m.add_function(wrap_pyfunction!(constant, m)?)?;
    m.add_function(wrap_pyfunction!(intersect, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_submodule(m)?;
    Ok(())
}
