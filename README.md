Automatic sqlite3 ORM generation from a kinda-structured schema.

Lots of features are missing, but the ORM generated:
- is fully typed
- ... in a very straightforward way that almost any editor should understand
- is very simple to use and understand
- almost looks like a human wrote it
- has *no* runtime dependencies outside stdlib (barring my wonky flask conn hack, TODO remove that)

The whole point is to have something easy and reliable when you just want to toss something simple together. Which is exactly what I want to do 90% of the time I'm using sqlite.

Want foreign keys and indexes and joins? Go use sqlalchemy

(actually I think 1-key indexes would be good to add and pretty simple to specify)

codegen via the wonderful ast.unparse, which I anticipate thouroughly abusing going forward.
