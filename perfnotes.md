# perf

## L2_recurrant



### Init at

``` python
PERSISTx = True
L2_KERNEL = 1
L2_MEMDEPTH = no multiplier
agent.start(max_iters=5)
repro=0.10, point=0.40, branch=0.10, cross=0.40

in_data = [DataFrom('static/datasets/letters0.csv', normalize=True),
               DataFrom('static/datasets/letters1.csv', normalize=True),
               DataFrom('static/datasets/letters2.csv', normalize=True),
               DataFrom('static/datasets/letters3.csv', normalize=True),
               DataFrom('static/datasets/letters4.csv', normalize=True)]

```

#### run 1:
Total run time:

```
L3 LEARNED: del
(Last hit: 2050s ago)
Total iterations: 4875.025
 Avg try length: 6.495258998671802
Total learn hits: 6
 Avg time btwn learn hits: 359.3333333333333
 Avg learn hit length: 2.1666666666666665
Total encounters: 367
 Avg time btwn encounters: 5.604904632152588
Learned:
['as', 'is', 'or', 'if', 'in', 'del']


L3 LEARNED: and
(Last hit: 610s ago)

L3 LEARNED: pass
(Last hit: 304s ago)
Total iterations: 4875.0
 Avg try length: 5.899882051282051
Total learn hits: 2
 Avg time btwn learn hits: 457.0
 Avg learn hit length: 10.0
Total encounters: 412
 Avg time btwn encounters: 5.20873786407767
Learned:
['as', 'is', 'or', 'if', 'in', 'del', 'and', 'pass']


L3 LEARNED: not
(Last hit: 2152s ago)

L3 LEARNED: def
(Last hit: 1201s ago)
Total iterations: 4875.0
 Avg try length: 6.942282051282051
Total learn hits: 2
 Avg time btwn learn hits: 1676.5
 Avg learn hit length: 13.0
Total encounters: 259
 Avg time btwn encounters: 9.23938223938224
Learned:
['as', 'is', 'or', 'if', 'in', 'del', 'and', 'pass', 'not', 'def']


L3 LEARNED: for
(Last hit: 1669s ago)
Total iterations: 4875.0
 Avg try length: 6.744410256410257
Total learn hits: 1
 Avg time btwn learn hits: 1669.0
 Avg learn hit length: 29.0
Total encounters: 378
 Avg time btwn encounters: 7.14021164021164
Learned:
['as', 'is', 'or', 'if', 'in', 'del', 'and', 'pass', 'not', 'def', 'for']


L3 LEARNED: try
(Last hit: 3267s ago)
Total iterations: 4875.0
 Avg try length: 7.160148717948718
Total learn hits: 1
 Avg time btwn learn hits: 3267.0
 Avg learn hit length: 32.0
Total encounters: 326
 Avg time btwn encounters: 10.220858895705522
Learned:
['as', 'is', 'or', 'if', 'in', 'del', 'and', 'pass', 'not', 'def', 'for', 'try']
```

## L2_ex_recurrant @ memdepth 2

### Init at

``` python
PERSISTx = True
L2_KERNEL = 1
L2_MEMDEPTH = 2
agent.start(max_iters=2)
repro=0.10, point=0.40, branch=0.10, cross=0.40
```

#### run 1 / 2
Total run time:

L3 LEARNED: as
(Last hit: 47s ago)

L3 LEARNED: or
(Last hit: 0s ago)

L3 LEARNED: in
(Last hit: 4s ago)

L3 LEARNED: is
(Last hit: 7s ago)

L3 LEARNED: if
(Last hit: 0s ago)

L3 LEARNED: del
(Last hit: 192s ago)

L3 LEARNED: def
(Last hit: 19s ago)

L3 LEARNED: from
(Last hit: 144s ago)

L3 LEARNED: and
(Last hit: 1140s ago)
Total iterations: 4874.025
 Avg try length: 6.395945855837835
Total learn hits: 9
 Avg time btwn learn hits: 172.55555555555554
 Avg learn hit length: 2.5555555555555554
Total encounters: 407
 Avg time btwn encounters: 4.176904176904177
Learned:
['as', 'or', 'in', 'is', 'if', 'del', 'def', 'from', 'and']


L3 LEARNED: not
(Last hit: 962s ago)

L3 LEARNED: for
(Last hit: 434s ago)
Total iterations: 4875.0
 Avg try length: 5.636805128205128
Total learn hits: 2
 Avg time btwn learn hits: 698.0
 Avg learn hit length: 14.5
Total encounters: 450
 Avg time btwn encounters: 5.815555555555555
Learned:
['as', 'or', 'in', 'is', 'if', 'del', 'def', 'from', 'and', 'not', 'for']

#### run 2 / 3:
Total run time:

L3 LEARNED: as
(Last hit: 92s ago)

L3 LEARNED: or
(Last hit: 0s ago)

L3 LEARNED: in
(Last hit: 114s ago)

L3 LEARNED: if
(Last hit: 94s ago)

L3 LEARNED: is
(Last hit: 3s ago)
Total iterations: 4874.025
 Avg try length: 7.068936864295936
Total learn hits: 5
 Avg time btwn learn hits: 60.6
 Avg learn hit length: 2.0
Total encounters: 136
 Avg time btwn encounters: 15.235294117647058
Learned:
['as', 'or', 'in', 'if', 'is']

Total iterations: 4875.0
 Avg try length: 6.782784615384616
Total learn hits: 0
 Avg time btwn learn hits: 0
 Avg learn hit length: 0
Total encounters: 139
 Avg time btwn encounters: 16.51798561151079
Learned:
['as', 'or', 'in', 'if', 'is']

#### run 4:

L3 LEARNED: or
(Last hit: 82s ago)

L3 LEARNED: in
(Last hit: 148s ago)

L3 LEARNED: as
(Last hit: 69s ago)

L3 LEARNED: if
(Last hit: 35s ago)

L3 LEARNED: is
(Last hit: 71s ago)

L3 LEARNED: and
(Last hit: 96s ago)

L3 LEARNED: try
(Last hit: 29s ago)
Total iterations: 4874.025
 Avg try length: 6.6436107734367384
Total learn hits: 7
 Avg time btwn learn hits: 75.71428571428571
 Avg learn hit length: 2.2857142857142856
Total encounters: 145
 Avg time btwn encounters: 17.20689655172414
Learned:
['or', 'in', 'as', 'if', 'is', 'and', 'try']


## L2_ex_recurrant @ memdepth 3

### Init at

``` python
PERSISTx = True
L2_KERNEL_MODE = 1
L2_MEMDEPTH = 3
L2_MAX_DEPTH = 6    # 10 is max, per Karoo user manual; has perf affect
L2_GAIN = .75       # A measure of the fit/random variance in the gene pool
L2_MEMDEPTH = 3     # Agent's "recurrent" memory, a multiple of L1's input sz
L2_MAX_POP = 40     # Genetic population sz; has perf affect
L2_TOURNYSZ = int(L2_MAX_POP * .25)  # Random genetic pool sz
agent.start(max_iters=5)
repro=0.10, point=0.40, branch=0.10, cross=0.40
```
#### run 1 - 5
L3 LEARNED: as
(Last hit: 0s ago)

L3 LEARNED: in
(Last hit: 8s ago)

L3 LEARNED: is
(Last hit: 10s ago)

L3 LEARNED: if
(Last hit: 7s ago)

L3 LEARNED: not
(Last hit: 38s ago)

L3 LEARNED: or
(Last hit: 37s ago)

L3 LEARNED: def
(Last hit: 1121s ago)
Total iterations: 4873.05
 Avg try length: 7.017930249022686
Total learn hits: 7
 Avg time btwn learn hits: 174.42857142857142
 Avg learn hit length: 2.2857142857142856
Total encounters: 403
 Avg time btwn encounters: 5.200992555831266
Learned:
['as', 'in', 'is', 'if', 'not', 'or', 'def']


L3 LEARNED: and
(Last hit: 2777s ago)

L3 LEARNED: for
(Last hit: 56s ago)
Total iterations: 4875.0
 Avg try length: 6.269748717948718
Total learn hits: 2
 Avg time btwn learn hits: 1416.5
 Avg learn hit length: 11.0
Total encounters: 408
 Avg time btwn encounters: 4.852941176470588
Learned:
['as', 'in', 'is', 'if', 'not', 'or', 'def', 'and', 'for']


L3 LEARNED: del
(Last hit: 1204s ago)

L3 LEARNED: try
(Last hit: 1333s ago)
Total iterations: 4875.0
 Avg try length: 6.149133333333333
Total learn hits: 2
 Avg time btwn learn hits: 1268.5
 Avg learn hit length: 14.0
Total encounters: 418
 Avg time btwn encounters: 4.636363636363637
Learned:
['as', 'in', 'is', 'if', 'not', 'or', 'def', 'and', 'for', 'del', 'try']

Total iterations: 4875.0
 Avg try length: 7.420123076923077
Total learn hits: 0
 Avg time btwn learn hits: 0
 Avg learn hit length: 0
Total encounters: 263
 Avg time btwn encounters: 8.463878326996198
Learned:
['as', 'in', 'is', 'if', 'not', 'or', 'def', 'and', 'for', 'del', 'try']


L3 LEARNED: pass
(Last hit: 3979s ago)
Total iterations: 4875.0
 Avg try length: 6.321476923076923
Total learn hits: 1
 Avg time btwn learn hits: 3979.0
 Avg learn hit length: 32.0
Total encounters: 330
 Avg time btwn encounters: 5.927272727272728
Learned:
['as', 'in', 'is', 'if', 'not', 'or', 'def', 'and', 'for', 'del', 'try', 'pass']


## L2_ex_recurrant @ memdepth 2 LONGRUN

### Init at

``` python
PERSISTx = True
L2_KERNEL_MODE = 1
L2_MEMDEPTH = 2
L2_MAX_DEPTH = 6    # 10 is max, per Karoo user manual; has perf affect
L2_GAIN = .75       # A measure of the fit/random variance in the gene pool
L2_MEMDEPTH = 3     # Agent's "recurrent" memory, a multiple of L1's input sz
L2_MAX_POP = 50     # Genetic population sz; has perf affect
L2_TOURNYSZ = int(L2_MAX_POP * .25)  # Random genetic pool sz
agent.start(max_iters=15)
repro=0.10, point=0.40, branch=0.10, cross=0.40
```
#### run 1 - 15