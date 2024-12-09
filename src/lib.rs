use std::{borrow::Cow, collections::HashMap, hash::Hash, sync::{Arc, LazyLock}};

use parking_lot::{ArcMutexGuard, Mutex, RawMutex};
use pyo3::{exceptions::{PyKeyError, PyRuntimeError, PyValueError}, intern, prelude::*, types::{PyBytes, PyType}};
use tribles::{id::IdOwner, prelude::*, query::{constantconstraint::ConstantConstraint, Binding, Constraint, ContainsConstraint, Query, TriblePattern, Variable}, value::{schemas::UnknownValue, RawValue}};

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

static TYPE_TO_ENTITY: LazyLock<Mutex<HashMap<PyPtrIdentity<PyType>, Id>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

static CONVERTERS: LazyLock<Mutex<HashMap<(Id, Id), Py<PyAny>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

#[pyfunction]
pub fn register_type(type_id: PyRef<'_, PyId>, typ: Py<PyType>) {
    let mut type_to_entity = TYPE_TO_ENTITY.lock();
    type_to_entity.insert(PyPtrIdentity(typ), type_id.0);
}

#[pyfunction]
pub fn register_converter(schema_id: PyRef<'_, PyId>, typ: Py<PyType>, converter: Py<PyAny>) -> PyResult<()> {
    let type_id = {
        let type_to_entity = TYPE_TO_ENTITY.lock();
        let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
            return Err(PyErr::new::<PyKeyError, _>("type should be registered first"));
        };
        entity.clone()
    };
    let mut converters = CONVERTERS.lock();
    converters.insert((schema_id.0, type_id), converter);
    Ok(())
}

#[pyclass(frozen, name = "Id")]
pub struct PyId(Id);

#[pymethods]
impl PyId {
    #[new]
    fn new(bytes: &[u8]) -> Result<Self, PyErr> {
        let Ok(id) = bytes.try_into() else {
            return Err(PyValueError::new_err("ids should be 16 bytes"));
        };
        let Some(id) = Id::new(id) else {
            return Err(PyValueError::new_err("id must be non-nil (contain non-zero bytes)"));
        };
        Ok(PyId(id))
    }

    #[staticmethod]
    pub fn hex(hex: &str) -> Result<Self, PyErr> {
        let Ok(id) = <[u8; 16]>::from_hex(hex) else {
            return Err(PyValueError::new_err("failed to parse hex id"));
        };
        let Some(id) = Id::new(id) else {
            return Err(PyValueError::new_err("id must be non-nil (contain non-zero bytes)"));
        };
        Ok(PyId(id))
    }

    pub fn bytes(&self) -> Cow<[u8]> {
        Cow::Borrowed(self.0.as_ref())
    }

    pub fn to_hex(&self) -> String {
        hex::encode_upper(self.0)
    }
}

#[pyclass(name = "IdOwner")]
pub struct PyIdOwner(Arc<Mutex<IdOwner>>);

#[pymethods]
impl PyIdOwner {
    #[new]
    fn new() -> Self {
        PyIdOwner(Arc::new(Mutex::new(IdOwner::new())))
    }

    pub fn rngid(&mut self) -> PyId {
        let owned_id = rngid();
        let id = self.0.lock_arc().insert(owned_id);
        PyId(id)
    }

    pub fn ufoid(&mut self) -> PyId {
        let owned_id = ufoid();
        let id = self.0.lock_arc().insert(owned_id);
        PyId(id)
    }

    pub fn fucid(&mut self) -> PyId {
        let owned_id = fucid();
        let id = self.0.lock_arc().insert(owned_id);
        PyId(id)
    }

    pub fn has(&self, v: PyRef<'_, PyVariable>) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.lock_arc().has(Variable::new(v.index)))
        }
    }

    pub fn owns(&mut self, id: PyRef<'_, PyId>) -> bool {
        self.0.lock_arc().owns(&id.0)
    }

    pub fn lock(&mut self) -> PyIdOwnerGuard {
        PyIdOwnerGuard(Some(self.0.lock_arc()))
    }
}

#[pyclass(name = "IdOwnerGuard")]
pub struct PyIdOwnerGuard(Option<ArcMutexGuard<RawMutex, IdOwner>>);

#[pymethods]
impl PyIdOwnerGuard {
    pub fn rngid(&mut self) -> PyResult<PyId> {
        let owned_id = rngid();
        if let Some(guard) = &mut self.0 {
            let id = guard.insert(owned_id);
            Ok(PyId(id))   
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn ufoid(&mut self) -> PyResult<PyId> {
        let owned_id = ufoid();
        if let Some(guard) = &mut self.0 {
            let id = guard.insert(owned_id);
            Ok(PyId(id))   
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn fucid(&mut self) -> PyResult<PyId> {
        let owned_id = fucid();
        if let Some(guard) = &mut self.0 {
            let id = guard.insert(owned_id);
            Ok(PyId(id))   
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn has(&mut self, v: PyRef<'_, PyVariable>) -> PyResult<PyConstraint> {
        if let Some(guard) = &mut self.0 {
            Ok(PyConstraint {
                constraint: Arc::new(guard.has(Variable::new(v.index)))
            })
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn owns(&mut self, id: PyRef<'_, PyId>) -> PyResult<bool> {
        if let Some(guard) = &mut self.0 {
            Ok(guard.owns(&id.0))
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn release(&mut self) {
        self.0 = None;
    }

    pub fn __exit__(&mut self) {
        self.release();
    }
}

#[pyclass(frozen, name = "Value")]
pub struct PyValue {
    value: RawValue,
    _value_schema: Id,
    _blob_schema: Option<Id>
}

#[pymethods]
impl PyValue {
    #[new]
    #[pyo3(signature = (bytes, value_schema, blob_schema=None))]
    fn new(bytes: &[u8], value_schema: PyRef<'_, PyId>, blob_schema: Option<PyRef<'_, PyId>>) -> PyResult<Self> {
        let Ok(bytes) = bytes.try_into() else {
            return Err(PyErr::new::<PyRuntimeError, _>("values should be 32 bytes"));
        };
        let value_schema = value_schema.0;
        let blob_schema = blob_schema.map(|s| s.0);

        Ok(PyValue {
            value: bytes,
            _value_schema: value_schema,
            _blob_schema: blob_schema,
        })
    }

    #[staticmethod]
    fn of(py: Python<'_>, value_schema: PyRef<'_, PyId>, value: Bound<'_, PyAny>) -> PyResult<Self> {
        let value_schema = value_schema.0;
        let type_id = {
            let typ = value.get_type().unbind();
            let type_to_entity = TYPE_TO_ENTITY.lock();
            let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
                return Err(PyErr::new::<PyKeyError, _>("type should be registered first"));
            };
            entity.clone()
        };
        let converters = CONVERTERS.lock();
        let Some(converter) = converters.get(&(value_schema, type_id)) else {
            return Err(PyErr::new::<PyKeyError, _>("converter should be registered first"));
        };
        let bytes = converter.call_method(py, intern!(py, "pack"), (value, ), None)?;
        let bytes = bytes.downcast_bound::<PyBytes>(py)?;
        let value: RawValue = bytes.as_bytes().try_into()?;
        Ok(Self {
            value,
            _value_schema: value_schema,
            _blob_schema: None,
        })
    }

    fn to(&self, py: Python<'_>, typ: Py<PyType>) -> PyResult<Py<PyAny>> {
        let type_id = {
            let type_to_entity = TYPE_TO_ENTITY.lock();
            let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
                return Err(PyErr::new::<PyKeyError, _>("type should be registered first"));
            };
            entity.clone()
        };
        let converters = CONVERTERS.lock();
        let Some(converter) = converters.get(&(self._value_schema, type_id)) else {
            return Err(PyErr::new::<PyKeyError, _>("converter should be registered first"));
        };
        let bytes = PyBytes::new(py, &self.value);
        converter.call_method(py, intern!(py, "unpack"), (bytes,), None)
    }

    pub fn value_schema(&self) -> PyId {
        PyId(self._value_schema)
    }

    pub fn blob_schema(&self) -> Option<PyId> {
        self._blob_schema.map(|s| 
            PyId(s))
    }

    pub fn is_handle(&self) -> bool {
        self._blob_schema.is_some()
    }

    pub fn bytes(&self) -> Cow<[u8]> {
        (&self.value).into()
    }
}

#[pyclass(name = "TribleSet")]
pub struct PyTribleSet(TribleSet);

#[pymethods]
impl PyTribleSet {
    #[staticmethod]
    pub fn empty() -> Self {
        PyTribleSet(TribleSet::new())
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

    pub fn add(&mut self, owner: Bound<'_, PyIdOwnerGuard>, e: Bound<'_, PyId>,  a: Bound<'_, PyId>,  v: Py<PyValue>) -> PyResult<()> {
        if !owner.borrow_mut().owns(e.borrow())? {
            return Err(PyErr::new::<PyRuntimeError, _>("can only add tribles with owned entity id"));
        };

        self.0.insert(&(Trible::new(OwnedId::transmute_force(&e.borrow().0), &a.borrow().0, &Value::<UnknownValue>::new(v.get().value))));
        Ok(())
    }

    pub fn consume(&mut self, other: &Bound<'_, Self>) {
        let set = &mut self.0;
        let other_set = std::mem::replace(&mut other.borrow_mut().0, TribleSet::new());
        set.union(other_set);
    }

    pub fn pattern(&self, ev: PyRef<'_, PyVariable>, av: PyRef<'_, PyVariable>, vv: PyRef<'_, PyVariable>) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.pattern(Variable::new(ev.index), Variable::new(av.index), Variable::<UnknownValue>::new(vv.index)))
        }
    }
}

#[pyclass(frozen, name = "Variable")]
pub struct PyVariable {
    index: u8,
    _value_schema: Id,
    _blob_schema: Option<Id>
}

#[pyclass(name = "Query")]
pub struct PyQuery {
    query: Mutex<Query<Arc<dyn Constraint<'static> + Send + Sync>, Box<dyn Fn(&Binding) -> Vec<PyValue> + Send>, Vec<PyValue>>>
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
        Value::<UnknownValue>::new(constant.get().value)));

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
pub fn solve(projected: Vec<Py<PyVariable>> ,constraint: &Bound<'_, PyConstraint>) -> PyQuery {
    let constraint = constraint.get().constraint.clone();

    let postprocessing = Box::new(move |binding: &Binding| {
        let mut vec = vec![];
        for v in &projected {
            let v = v.get();
            let value = *binding.get(v.index).expect("constraint should contain projected variables");
            vec.push(PyValue {
                value,
                _value_schema: v._value_schema,
                _blob_schema: v._blob_schema
            });
        }
        vec
    }) as Box<dyn Fn(&Binding) -> Vec<PyValue> + Send>;

    let query = tribles::query::Query::new(constraint, postprocessing);

    PyQuery {
        query: Mutex::new(query)
    }
}

#[pymethods]
impl PyQuery {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Vec<PyValue>> {
        slf.query.get_mut().next()
    }
}

/// The `tribles` python module.
#[pymodule]
#[pyo3(name = "tribles")]
pub fn tribles_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTribleSet>()?;
    m.add_class::<PyId>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyVariable>()?;
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
