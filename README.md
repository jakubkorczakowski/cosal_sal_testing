# Framework workflow

Dataset requirements:
- it has to have data separated into classes (categories).

Example testing datasets: CoSAL2015, CoSOD3k or CoCA.
Usually saliency algorithms use data that isn't separated into classes.
We provide scripts to flatten directory structure and bring it back to original form.


1. Prepare saliency maps for your algorithm. Store it in `Data` directory defined in `config.py` file.
2. Run
```
    python3 src/main.py
```

ddt -> simplecrf -\
                   >-> merge_maps
saliency_results -/


# TODO

1. (Docs) Add proper readme.md
    - description
    - how to run (add coarse requirements)
    - results
    - 
2. (Code) Add parse args instead of constants.
    - clean up `co_locate` method
