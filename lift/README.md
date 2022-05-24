LIFT
----

Backends:

Use RLgraph as the default backend with TensorFlow, e.g.:

```
pip install tensorflow rlgraph
```

Optimisation Baselines:

Install OpenTuner in a separate repo:

```
git clone https://github.com/jansel/opentuner
```

N.b.: RLgraph works with Python 3.x, OpenTuner's install
requirements are for 2.7, even though the master works with 3.x (pip version
does not). In the requirements.txt, simply remove the entry "pysqlite>=2.6.3
" as that will cause an installation failure with Python 3.x. After removing 
the entry, execute:
```
pip install -r requirements.txt
```