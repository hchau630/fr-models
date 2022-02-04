# fr-models
A python package for simulating and fitting firing rate models using PyTorch backend

## Speed
GPU only speeds up computations when the model is sufficiently large. Here is some data for the total time taken for setting up a model and then computing the response for an input after 500.0xtau seconds, for 10 repeated runs:

Neurons size: (2,101)
- GPU: [5.085921160876751, 0.7013185545802116, 0.6913328617811203, 0.7159170210361481, 0.7139800786972046, 0.7154168039560318, 0.7100193127989769, 0.7038960978388786, 0.6892224326729774, 0.7034369707107544]
- CPU (torch): [0.41797085106372833, 0.4170943573117256, 0.3905418589711189, 0.40386854112148285, 0.39362432807683945, 0.3915015086531639, 0.41379377990961075, 0.4049643278121948, 0.4000104144215584, 0.4006969705224037]
- CPU (scipy): [0.18213913589715958, 0.15131710469722748, 0.14938997477293015, 0.1478876918554306, 0.15159278362989426, 0.15775398164987564, 0.1561225950717926, 0.16125426441431046, 0.1587381362915039, 0.1507381796836853]

Neurons size: (2,1001)
- GPU: [5.3193482756614685, 0.720893956720829, 0.717757061123848, 0.7058169543743134, 0.7119910642504692, 0.6943261846899986, 0.6873603165149689, 0.6985301226377487, 0.6944873854517937, 0.6970896124839783]
- CPU (torch): [0.8981039524078369, 0.8265528902411461, 0.8224083930253983, 0.8374765962362289, 0.8501281589269638, 0.8528627157211304, 0.8202274441719055, 0.7856136038899422, 0.8093823343515396, 0.8215753585100174]
- CPU (scipy): [1.7840007916092873, 1.5026494711637497, 1.7460966259241104, 1.7720819935202599, 1.753230333328247, 1.7249890640377998, 2.1759914234280586, 1.5235988795757294, 1.6750506162643433, 1.8988630548119545]

Neurons size: (2,1501)
- GPU: [5.00920595228672, 0.7001928240060806, 0.69356519728899, 0.688030406832695, 0.6962978467345238, 0.6847635954618454, 0.693671740591526, 0.6843123137950897, 0.6863279044628143, 0.6838540807366371]
- CPU (torch): [2.440558783710003, 2.341607742011547, 2.610865719616413, 2.819392330944538, 2.5597817674279213, 2.1728768348693848, 2.4511165022850037, 2.36025907099247, 2.4153554439544678, 2.270886242389679]
- CPU (scipy): [7.038497306406498, 7.16134487837553, 5.985429897904396, 6.177450850605965, 6.011391706764698, 6.441509731113911, 7.290819466114044, 7.116204433143139, 6.007739707827568, 7.245858445763588]

As one can see, it is only at around 2001 neurons that the GPU begins to show its power.