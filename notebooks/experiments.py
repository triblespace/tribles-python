import marimo

__generated_with = "0.10.6"
app = marimo.App()


@app.cell
def _():
    import maturin_import_hook
    maturin_import_hook.install(settings=maturin_import_hook.MaturinSettings(release=True))
    import tribles
    return maturin_import_hook, tribles


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    print(sys.version)
    return (sys,)


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _():
    import os
    return (os,)


@app.cell
def _():
    import hifitime
    return (hifitime,)


@app.cell
def _():
    import time
    return (time,)


@app.cell
def _():
    import timeit
    return (timeit,)


@app.cell
def _():
    import fractions
    from fractions import Fraction
    return Fraction, fractions


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _(mo):
    mo.md("""# Tribles""")
    return


@app.cell
def _(tribles):
    TribleSet = tribles.TribleSet
    return (TribleSet,)


@app.cell
def _(tribles):
    Id = tribles.Id
    return (Id,)


@app.cell
def _(tribles):
    Value = tribles.Value
    return (Value,)


@app.cell
def _(tribles):
    IdOwner = tribles.IdOwner
    return (IdOwner,)


@app.cell
def _(IdOwner):
    owner = IdOwner()
    with owner.lock() as o:
        o.rngid()
    return o, owner


@app.cell
def _(TribleSet):
    context = TribleSet.empty()
    return (context,)


@app.cell
def _(FR256LE, Value, fractions):
    Value.of(fractions.Fraction(1,2), FR256LE).to(fractions.Fraction)
    return


@app.cell
def _(GenId, Value, Variable, tribles):
    class Namespace:
        def __init__(self, description, names = None):
            self.description = description
            self.names = names or tribles.get_label_names(description)
            self.name_to_value_schema = {
                name: tribles.get_value_schema(description, id)
                for (name, id) in self.names.items()
            }
            self.name_to_blob_schema = {
                name: tribles.get_blob_schema(description, id)
                for (name, id) in self.names.items()
            }

        def __entity(self, set, entity, owner):
            if "@id" in entity:
                entity_id = entity["@id"]
                if owner and not owner.owns(entity_id):
                    raise RuntimeError("entity id not owned by provided owner");
            else:
                entity_id = owner.fucid()

            for key, value in entity.items():
                if key == "@id":
                    continue
                attr_id = self.names[key]
                attr_value_schema = self.name_to_value_schema[key]
                attr_blob_schema = self.name_to_blob_schema[key]
                value = Value.of(value, attr_value_schema, attr_blob_schema)
                set.add(entity_id, attr_id, value)

            return set

        def entity(self, entity, owner = None):
            set = tribles.TribleSet.empty()
            
            self.__entity(set, entity, owner)

            return set

        def entities(self, entities, owner = None):
            set = tribles.TribleSet.empty()

            for entity in entities:
                self.__entity(set, entity, owner)

            return set

        def pattern(self, set, entities):
            def __with_ctx(ctx):
                constraints = []
                for entity in entities:
                    if "@id" in entity:
                        entity_id = entity["@id"]
                    else:
                        entity_id = ctx.new()
                    if type(entity_id) is Variable:
                        e_v = entity_id
                        e_v.annotate_schemas(GenId)
                    else:
                        e_v = ctx.new()
                        e_v.annotate_schemas(GenId)
                        constraints.append(
                            tribles.constant(
                                e_v.index,
                                Value.of(entity_id, GenId),
                            )
                        )
        
                    for key, value in entity.items():
                        if key == "@id":
                            continue
                        attr_id = self.names[key]
                        attr_value_schema = self.name_to_value_schema[key]
                        attr_blob_schema = self.name_to_blob_schema[key]
        
                        a_v = ctx.new()
                        a_v.annotate_schemas(GenId)
                        constraints.append(
                            tribles.constant(
                                a_v.index,
                                Value.of(attr_id, GenId),
                            )
                        )
        
                        if type(value) is Variable:
                            v_v = value
                            v_v.annotate_schemas(attr_value_schema, attr_blob_schema)
                        else:
                            v_v = ctx.new()
                            v_v.annotate_schemas(attr_value_schema, attr_blob_schema)
                            constraints.append(
                                tribles.constant(
                                    v_v.index, Value.of(value, attr_value_schema, attr_blob_schema)
                                )
                            )
                        constraints.append(
                            set.pattern(e_v.index, a_v.index, v_v.index)
                        )
                return tribles.intersect(constraints)
            return __with_ctx
    return (Namespace,)


@app.cell
def _(Id, Namespace, tribles):
    metadata_ns = Namespace(
        tribles.metadata_description(),
        {
            "attr_name": Id.hex("2E26F8BA886495A8DF04ACF0ED3ACBD4"),
            "value_schema": Id.hex("213F89E3F49628A105B3830BD3A6612C"),
            "blob_schema": Id.hex("02FAF947325161918C6D2E7D9DBA3485"),
        },
    )
    return (metadata_ns,)


@app.cell
def _(Variable):
    class VariableContext:
        def __init__(self):
            self.variables = []

        def new(self, name=None):
            i = len(self.variables)
            assert i < 128
            v = Variable(i, name)
            self.variables.append(v)
            return v

        def check_schemas(self):
            for v in self.variables:
                if not v.schema:
                    if v.name:
                        name = "'" + v.name + "'"
                    else:
                        name = "_"
                    raise TypeError(
                        "missing schema for variable "
                        + name
                        + "/"
                        + str(v.index)
                    )
    return (VariableContext,)


@app.cell
def _(VariableContext, tribles):
    def find(query):
        ctx = VariableContext()
        projected_variable_names = query.__code__.co_varnames[1:]
        projected_variables = [ctx.new(n) for n in projected_variable_names]
        constraint = query(*projected_variables)(ctx)
        ctx.check_schemas()
        projected_variable_schemas = [(v.index, v.schema) for v in projected_variables]
        for result in tribles.solve(projected_variable_schemas, constraint):
            yield tuple(result)
    return (find,)


@app.cell
def _(tribles):
    def concrete_type(entity_id):
        def inner(type):
            tribles.register_type(entity_id, type)
            return type
        return inner
    return (concrete_type,)


@app.cell
def _(tribles):
    def to_value(schema, type):
        def inner(converter):
            tribles.register_to_value_converter(schema, type, converter)
            return converter
        return inner
    return (to_value,)


@app.cell
def _(tribles):
    def from_value(schema, type):
        def inner(converter):
            tribles.register_from_value_converter(schema, type, converter)
            return converter
        return inner
    return (from_value,)


@app.cell
def _(Id, concrete_type, fractions):
    concrete_type(Id.hex("A75056BFA2AE677767B1DB8B01AFA322"))(int)
    concrete_type(Id.hex("7D06820D69947D76E7177E5DEA4EA773"))(str)
    concrete_type(Id.hex("BF11820EC384447B666988490D727A1C"))(Id)
    concrete_type(Id.hex("83D62F300ED37850DFFB42E6226117ED"))(fractions.Fraction)
    return


@app.cell
def _(Id):
    """an random 128 bit id (the first 128bits are zero padding)"""
    GenId = Id.hex("B08EE1D45EB081E8C47618178AFE0D81")
    return (GenId,)


@app.cell
def _(Id):
    """32 raw bytes"""
    UnknownBytes = Id.hex("4EC697E8599AC79D667C722E2C8BEBF4")
    return (UnknownBytes,)


@app.cell
def _(Id):
    """A \0 terminated short utf-8 string that fits of up to 32 bytes of characters"""
    ShortString = Id.hex("2D848DB0AF112DB226A6BF1A3640D019")
    return (ShortString,)


@app.cell
def _(Id):
    """an signed 256bit integer in big endian encoding"""
    I256BE = Id.hex("CE3A7839231F1EB390E9E8E13DAED782")
    return (I256BE,)


@app.cell
def _(Id):
    """a signed 256bit integer in little endian encoding"""
    I256LE = Id.hex("DB94325A37D96037CBFC6941A4C3B66D")
    return (I256LE,)


@app.cell
def _(Id):
    """an unsigned 256bit integer in big endian encoding"""
    U256BE = Id.hex("DC3CFB719B05F019FB8101A6F471A982")
    return (U256BE,)


@app.cell
def _(Id):
    """an unsigned 256bit integer in little endian encoding"""
    U256LE = Id.hex("49E70B4DBD84DC7A3E0BDDABEC8A8C6E")
    return (U256LE,)


@app.cell
def _(Id):
    """a time duration in nanoseconds stored as a signed 256bit big endian integer"""
    NSDuration = Id.hex("675A2E885B12FCBC0EEC01E6AEDD8AA8")
    return (NSDuration,)


@app.cell
def _(Id):
    """a unitless fraction stored as a (numerator, denominator) pair of signed 128bit little endian integers"""
    FR256LE = Id.hex("0A9B43C5C2ECD45B257CDEFC16544358")
    return (FR256LE,)


@app.cell
def _(GenId, to_value):
    @to_value(schema = GenId, type = str)
    def _convert(value):
        assert len(value) == 32
        return bytes.fromhex(value)
    return


@app.cell
def _(GenId, from_value):
    @from_value(schema = GenId, type = str)
    def _convert(bytes):
        return bytes.hex().upper()
    return


@app.cell
def _(GenId, Id, to_value):
    @to_value(schema = GenId, type = Id)
    def _convert(concrete):
        return bytes(16) + concrete.bytes()
    return


@app.cell
def _(GenId, Id, from_value):
    @from_value(schema = GenId, type = Id)
    def _convert(bytes):
        assert all(v == 0 for v in bytes[0: 16])
        assert not all(v == 0 for v in bytes[16: 32])
        return Id(bytes[16:32])
    return


@app.cell
def _(ShortString, to_value):
    @to_value(schema = ShortString, type = str)
    def _convert(concrete):
        b = bytes(concrete, 'utf-8')
        assert len(b) <= 32
        assert 0 not in b
        return b + bytes(32 - len(b))
    return


@app.cell
def _(ShortString, from_value):
    @from_value(schema = ShortString, type = str)
    def _convert(bytes):
        try:
            end = bytes.index(0)
            return bytes[0:end].decode('utf-8')
        except:
            return bytes.decode('utf-8')
    return


@app.cell
def _(I256BE, to_value):
    @to_value(schema = I256BE, type = int)
    def _convert(concrete):
        return concrete.to_bytes(32, byteorder='big', signed=True)
    return


@app.cell
def _(I256BE, from_value):
    @from_value(schema = I256BE, type = int)
    def _convert(bytes):
        return int.from_bytes(bytes, byteorder='big', signed=True)
    return


@app.cell
def _(I256LE, to_value):
    @to_value(schema = I256LE, type = int)
    def _convert(concrete):
        return concrete.to_bytes(32, byteorder='little', signed=True)
    return


@app.cell
def _(I256LE, from_value):
    @from_value(schema = I256LE, type = int)
    def _convert(bytes):
        return int.from_bytes(bytes, byteorder='little', signed=True)
    return


@app.cell
def _(U256BE, to_value):
    @to_value(schema = U256BE, type = int)
    def _convert(concrete):
        return concrete.to_bytes(32, byteorder='big', signed=False)
    return


@app.cell
def _(U256BE, from_value):
    @from_value(schema = U256BE, type = int)
    def _convert(bytes):
        return int.from_bytes(bytes, byteorder='big', signed=False)
    return


@app.cell
def _(U256LE, to_value):
    @to_value(schema = U256LE, type = int)
    def pack(concrete):
        return concrete.to_bytes(32, byteorder='little', signed=False)
    return (pack,)


@app.cell
def _(U256LE, from_value):
    @from_value(schema = U256LE, type = int)
    def _convert(bytes):
        return int.from_bytes(bytes, byteorder='little', signed=False)
    return


@app.cell
def _(NSDuration, to_value):
    @to_value(schema = NSDuration, type = int)
    def _convert(concrete):
        return concrete.to_bytes(32, byteorder='big', signed=False)
    return


@app.cell
def _(NSDuration, from_value):
    @from_value(schema = NSDuration, type = int)
    def _convert(bytes):
        return int.from_bytes(bytes, byteorder='big', signed=False)
    return


@app.cell
def _(FR256LE, fractions, to_value):
    @to_value(schema = FR256LE, type = fractions.Fraction)
    def _convert(concrete):
        n, d = concrete.as_integer_ratio()
        nb = n.to_bytes(16, byteorder='little', signed=True)
        db = d.to_bytes(16, byteorder='little', signed=True)
        return nb + db
    return


@app.cell
def _(FR256LE, fractions, from_value):
    @from_value(schema = FR256LE, type = fractions.Fraction)
    def _convert(bytes):
        n = int.from_bytes(bytes[0:16], byteorder='little', signed=True)
        d = int.from_bytes(bytes[16:32], byteorder='little', signed=True)
        return fractions.Fraction(n, d)
    return


@app.cell
def _(FR256LE, to_value):
    @to_value(schema = FR256LE, type = int)
    def _convert(concrete):
        nb = concrete.to_bytes(16, byteorder='little', signed=True)
        db = (1).to_bytes(16, byteorder='little', signed=True)
        return nb + db
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(metadata_ns):
    metadata_ns.names
    return


@app.cell
def _(
    FR256LE,
    GenId,
    Id,
    NSDuration,
    Namespace,
    ShortString,
    U256LE,
    metadata_ns,
):
    experiments = Namespace(
        metadata_ns.entities(
            [
                {
                    "@id": Id.hex("EC80E5FBDF856CD47347D1BCFB5E0D3E"),
                    "attr_name": "label",
                    "value_schema": ShortString,
                },
                {
                    "@id": Id.hex("E3ABE180BD5742D92616671E643FA4E5"),
                    "attr_name": "experiment",
                    "value_schema": GenId,
                },
                {
                    "@id": Id.hex("A8034B8D0D644DCAA053CA1374AE92A0"),
                    "attr_name": "element_count",
                    "value_schema": U256LE,
                },
                {
                    "@id": Id.hex("1C333940F98D0CFCEBFCC408FA35FF92"),
                    "attr_name": "cpu_time",
                    "value_schema": NSDuration,
                },
                {
                    "@id": Id.hex("999BF50FFECF9C0B62FD23689A6CA0D0"),
                    "attr_name": "wall_time",
                    "value_schema": NSDuration,
                },
                {
                    "@id": Id.hex("78D9B9230C044FA4E1585AFD14CFB3EE"),
                    "attr_name": "avg_distance",
                    "value_schema": FR256LE,
                },
                {
                    "@id": Id.hex("AD5DD3F72FA8DD67AF0D0DA5298A98B9"),
                    "attr_name": "change_count",
                    "value_schema": FR256LE,
                },
                {
                    "@id": Id.hex("2DB0F43553543173C42C8AE1573A38DB"),
                    "attr_name": "layer_explored",
                    "value_schema": FR256LE,
                },
            ]
        )
    )
    return (experiments,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""# Benchmarks""")
    return


@app.cell
def _(Fraction, experiments, owner):
    def gen_data(size):
        for i in range(size):
            yield experiments.entity({
                "layer_explored": i,
                "cpu_time": 500 * i,
                "wall_time": 600 * i,
                "avg_distance": Fraction(i, 1)}, owner)
    return (gen_data,)


@app.cell
def _(TribleSet, gen_data):
    def bench_consume(size):
        set = TribleSet.empty()
        for entity in gen_data(size):
            set.consume(entity)
        return set
    return (bench_consume,)


@app.cell
def _(TribleSet, gen_data):
    def bench_mutable_add(size):
        set = TribleSet.empty()
        for entity in gen_data(size):
            set += entity
        return set
    return (bench_mutable_add,)


@app.cell
def _(TribleSet, gen_data):
    def bench_sum(size):
        set = sum(gen_data(size), start = TribleSet.empty())
        return set
    return (bench_sum,)


@app.cell
def _(timeit):
    def time_ns(l):
        s = timeit.timeit(l, number=1)
        return int(s * 1e9)
    return (time_ns,)


@app.cell
def _(mo):
    mo.md("""### Insert""")
    return


@app.cell
def _(metadata_ns):
    metadata_ns.names
    return


@app.cell
def _(metadata_ns):
    len(metadata_ns.description)
    return


@app.cell
def _(metadata_ns):
    metadata_ns.description or 1
    return


@app.cell
def _(FR256LE, GenId, Id, NSDuration, ShortString, U256LE, metadata_ns):
    len(metadata_ns.entities(
        [
            {
                "@id": Id.hex("EC80E5FBDF856CD47347D1BCFB5E0D3E"),
                "attr_name": "label",
                "value_schema": ShortString,
            },
            {
                "@id": Id.hex("E3ABE180BD5742D92616671E643FA4E5"),
                "attr_name": "experiment",
                "value_schema": GenId,
            },
            {
                "@id": Id.hex("A8034B8D0D644DCAA053CA1374AE92A0"),
                "attr_name": "element_count",
                "value_schema": U256LE,
            },
            {
                "@id": Id.hex("1C333940F98D0CFCEBFCC408FA35FF92"),
                "attr_name": "cpu_time",
                "value_schema": NSDuration,
            },
            {
                "@id": Id.hex("999BF50FFECF9C0B62FD23689A6CA0D0"),
                "attr_name": "wall_time",
                "value_schema": NSDuration,
            },
            {
                "@id": Id.hex("78D9B9230C044FA4E1585AFD14CFB3EE"),
                "attr_name": "avg_distance",
                "value_schema": FR256LE,
            },
            {
                "@id": Id.hex("AD5DD3F72FA8DD67AF0D0DA5298A98B9"),
                "attr_name": "change_count",
                "value_schema": FR256LE,
            },
            {
                "@id": Id.hex("2DB0F43553543173C42C8AE1573A38DB"),
                "attr_name": "layer_explored",
                "value_schema": FR256LE,
            },
        ]
    ))
    return


@app.cell
def _(bench_consume, element_count_exp, experiments, owner, time_ns):
    _experiment = owner.fucid()
    bench_insert_consume_data = experiments.entity(
        {"@id": _experiment, "label": "consume"}, owner
    )

    for _i in range(element_count_exp):
        bench_insert_consume_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_consume(2**_i)),
                "element_count": (2**_i) * 4,
            },
            owner
        )
    return (bench_insert_consume_data,)


@app.cell
def _(bench_mutable_add, element_count_exp, experiments, owner, time_ns):
    _experiment = owner.fucid()
    bench_insert_mutable_add_data = experiments.entity(
        {"@id": _experiment, "label": "mutable_add"},
        owner
    )
    for _i in range(element_count_exp):
        sample = owner.fucid()
        bench_insert_mutable_add_data += experiments.entity(
            {
                "@id": sample,
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_mutable_add(2**_i)),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_insert_mutable_add_data, sample


@app.cell
def _(bench_sum, element_count_exp, experiments, owner, time_ns):
    _experiment = owner.fucid()
    bench_insert_sum_data = experiments.entity({"@id": _experiment, "label": "sum"})
    for _i in range(element_count_exp):
        bench_insert_sum_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_sum(2**_i)),
                "element_count": (2**_i) * 4,
            },
            owner
        )
    return (bench_insert_sum_data,)


@app.cell
def _(mo):
    mo.md("""### Query""")
    return


@app.cell
def _(bench_consume, experiments, find, time_ns):
    def bench_query_find(size):
        set = bench_consume(size)
        return time_ns(
            lambda: sum(
                [
                    1
                    for _ in find(
                        lambda layer, cpu, wall, distance: experiments.pattern(
                            set,
                            [
                                {
                                    "layer_explored": layer,
                                    "cpu_time": cpu,
                                    "wall_time": wall,
                                    "avg_distance": distance,
                                }
                            ],
                        )
                    )
                ]
            )
        )
    return (bench_query_find,)


@app.cell
def _(bench_query_find, element_count_exp, experiments, owner):
    _experiment = owner.fucid()
    bench_query_find_data = experiments.entity({"@id": _experiment, "label": "find"})
    for _i in range(element_count_exp):
        bench_query_find_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": bench_query_find(2**_i),
                "element_count": (2**_i) * 4,
            }
        )
    return (bench_query_find_data,)


@app.cell
def _(mo):
    mo.md("""## RDFLib""")
    return


@app.cell
def _():
    from rdflib import Graph, URIRef, BNode, Literal, Namespace as RDFNamespace
    return BNode, Graph, Literal, RDFNamespace, URIRef


@app.cell
def _():
    from rdflib.plugins import sparql
    return (sparql,)


@app.cell
def _(RDFNamespace):
    benchns = RDFNamespace("http://example.org/benchmark/")
    rdf_layer_explored = benchns.layer_explored
    rdf_avg_cpu_time = benchns.avg_cpu_time
    rdf_avg_wall_time = benchns.avg_wall_time
    rdf_avg_distance = benchns.avg_distance
    return (
        benchns,
        rdf_avg_cpu_time,
        rdf_avg_distance,
        rdf_avg_wall_time,
        rdf_layer_explored,
    )


@app.cell
def _(mo):
    mo.md("""### Insert""")
    return


@app.cell
def _(BNode, Fraction, Graph, Literal, benchns):
    def bench_rdf(n):
        g = Graph()
        g.bind("benchmark", benchns)

        for i in range(n):
            eid = BNode()  # a GUID is generated
            g.add((eid, benchns.layer_explored, Literal(i)))
            g.add((eid, benchns.avg_cpu_time, Literal(500 * i)))
            g.add((eid, benchns.avg_wall_time, Literal(600 * i)))
            g.add((eid, benchns.avg_distance, Literal(Fraction(i, 1))))

        return g
    return (bench_rdf,)


@app.cell
def _(bench_rdf, element_count_exp, experiments, owner, time_ns):
    _experiment = owner.fucid()
    bench_insert_rdf_data = experiments.entity({"@id": _experiment, "label": "RDFLib"})
    for _i in range(element_count_exp):
        bench_insert_rdf_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_rdf(2**_i)),
                "element_count": (2**_i) * 4,
            },
            owner
        )
    return (bench_insert_rdf_data,)


@app.cell
def _(mo):
    mo.md("""### Query""")
    return


@app.cell
def _(bench_rdf, time_ns):
    def bench_rdf_query_adhoc(n):
        g = bench_rdf(n)

        query = """
        SELECT ?layer ?cpu ?wall ?distance
        WHERE {
            ?a benchmark:layer_explored ?layer;
               benchmark:avg_cpu_time ?cpu;
               benchmark:avg_wall_time ?wall;
               benchmark:avg_distance ?distance .
        }"""
        return time_ns(lambda: sum([1 for _ in g.query(query)]))
    return (bench_rdf_query_adhoc,)


@app.cell
def _(bench_rdf, benchns, sparql, time_ns):
    _prepared_query = sparql.prepareQuery(
        """
        SELECT ?layer ?cpu ?wall ?distance
        WHERE {
            ?a benchmark:layer_explored ?layer;
               benchmark:avg_cpu_time ?cpu;
               benchmark:avg_wall_time ?wall;
               benchmark:avg_distance ?distance .
        }""",
        initNs = { "benchmark": benchns })

    def bench_rdf_query_prepared(n):
        g = bench_rdf(n) 
        return time_ns(lambda: sum([1 for _ in g.query(_prepared_query)]))
    return (bench_rdf_query_prepared,)


@app.cell
def _(bench_rdf_query_adhoc, element_count_exp, experiments, owner):
    _experiment = owner.fucid()
    bench_query_adhoc_rdf_data = experiments.entity({"@id": _experiment, "label": "RDFLib (adhoc)"})
    for _i in range(element_count_exp):
        bench_query_adhoc_rdf_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": bench_rdf_query_adhoc(2**_i),
                "element_count": (2**_i) * 4,
            },
            owner
        )
    return (bench_query_adhoc_rdf_data,)


@app.cell
def _(bench_rdf_query_prepared, element_count_exp, experiments, owner):
    _experiment = owner.fucid()
    bench_query_prepared_rdf_data = experiments.entity({"@id": _experiment, "label": "RDFLib (prepared)"})
    for _i in range(element_count_exp):
        bench_query_prepared_rdf_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": bench_rdf_query_prepared(2**_i),
                "element_count": (2**_i) * 4,
            },
            owner
        )
    return (bench_query_prepared_rdf_data,)


@app.cell
def _(mo):
    mo.md("""## Polars""")
    return


@app.cell
def _(pl):
    df_triples = pl.DataFrame(
        {
            "e": ["a", "b", "c"],
            "a": [1, 2, 2],
            "v": [100, 200, 300],
        }
    )
    df_triples
    return (df_triples,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""## Evaluation""")
    return


@app.cell
def _():
    element_count_exp = 1
    return (element_count_exp,)


@app.cell
def _(
    bench_insert_consume_data,
    bench_insert_mutable_add_data,
    bench_insert_rdf_data,
    bench_insert_sum_data,
):
    bench_insert_data = bench_insert_consume_data + bench_insert_mutable_add_data + bench_insert_sum_data + bench_insert_rdf_data
    return (bench_insert_data,)


@app.cell
def _(Id, alt, bench_insert_data, experiments, find, mo):
    benchdata = alt.Data(
        values=[
            {
                "label": l.to(str),
                "time/fact (ns)": t.to(int) / c.to(int),
                "#facts": c.to(int),
            }
            for e, l, t, c in find(
                lambda ctx, e, l, t, c: experiments.pattern(ctx, bench_insert_data,
                    [
                        {"experiment": e, "wall_time": t, "element_count": c},
                        {Id: e, "label": l},
                    ],
                )
            )
        ]
    )

    # Create an Altair chart
    benchchart = (
        alt.Chart(benchdata)
        .mark_line()
        .encode(
            x=alt.Y("#facts:Q" ).scale(type="log"),
            y=alt.Y("time/fact (ns):Q").scale(type="log"),
            color="label:O",
        )
    )

    # Make it reactive ⚡
    benchchart = mo.ui.altair_chart(benchchart)
    return benchchart, benchdata


@app.cell
def _(benchchart, mo):
    mo.vstack([benchchart, benchchart.value.head()])
    return


@app.cell
def _(
    bench_query_adhoc_rdf_data,
    bench_query_find_data,
    bench_query_prepared_rdf_data,
):
    bench_query_data = bench_query_find_data + bench_query_adhoc_rdf_data + bench_query_prepared_rdf_data
    return (bench_query_data,)


@app.cell
def _(Id, alt, bench_query_data, experiments, find, mo):
    benchdata_query = alt.Data(
        values=[
            {
                "label": l.to(str),
                "time/fact (ns)": t.to(int) / c.to(int),
                "#facts": c.to(int),
            }
            for e, l, t, c in find(
                lambda ctx, e, l, t, c: experiments.pattern(
                    ctx,
                    bench_query_data,
                    [
                        {"experiment": e, "wall_time": t, "element_count": c},
                        {Id: e, "label": l},
                    ],
                )
            )
        ]
    )

    # Create an Altair chart
    benchchart_query = (
        alt.Chart(benchdata_query)
        .mark_line()
        .encode(
            x=alt.Y("#facts:Q" ).scale(type="log"),
            y=alt.Y("time/fact (ns):Q").scale(type="log"),
            color="label:O",
        )
    )

    # Make it reactive ⚡
    benchchart_query = mo.ui.altair_chart(benchchart_query)
    return benchchart_query, benchdata_query


@app.cell
def _(benchchart_query, mo):
    mo.vstack([benchchart_query, benchchart_query.value.head()])
    return


@app.cell
def _(mo):
    mo.md("""# Small Worlds""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## dataset
         * distances (nbojects, dim)   f32 matrix    for tests objects
         * neighbors (nbobjects, nbnearest) int32 matrix giving the num of nearest neighbors in train data
          * test      (nbobjects, dim)   f32 matrix  test data
          * train     (nbobjects, dim)   f32 matrix  train data

        load hdf5 data file benchmarks from https://github.com/erikbern/ann-benchmarks
        """
    )
    return


@app.cell
def _(e):
    e.avg_cpu_time
    return


@app.cell
def _(e):
    e.avg_distance
    return


if __name__ == "__main__":
    app.run()
