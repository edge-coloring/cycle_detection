# Cycle detection program to confirm Subsection 9.1

`check_safe.py` is the implementation of the pseudocode shown in Appendix C.4.
Provide a `.conf` file and the contraction edge set (delimited by `+` symbols) like the following:

```
python3 check_safe.py path/to/conf 3+5+8+13
```

You can use `checkall.sh` to check for all configurations in a certain directory containing `conf/` and `summary.csv`.
For example, running

```
./checkall.sh ../projective_dual_configurations/reducible
```

will check all the configurations having contraction size 5 or more in the `reducible` directory.
