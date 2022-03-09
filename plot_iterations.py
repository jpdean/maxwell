# Plots the output from iterations.py.

import plotly.graph_objects as go
import pickle

save_figs_to_file = False

ndofs, times, iters = pickle.load(open("iteration_results.p", "rb"))

iter_fig = go.Figure(data=go.Scatter(x=ndofs, y=iters))
iter_fig.update_layout(width=500,
                       height=500,
                       font=dict(size=20),
                       xaxis=dict(title="NDOFs",
                                  type="log",
                                  exponentformat='power',
                                  dtick=1),
                       yaxis=dict(title="Iterations",
                                  rangemode="tozero"))
iter_fig.show()
