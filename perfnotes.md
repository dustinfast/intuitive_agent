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

## L2_recurrant

### Init at

``` python
PERSISTx = True
L2_KERNEL = 1
L2_MEMDEPTH = 2
agent.start(max_iters=5)
repro=0.10, point=0.40, branch=0.10, cross=0.40