import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omnetpp.scave import results, chart, utils, plot, vectorops as ops

#########################################################################
import sys
from IPython.utils import strdispatch
sys.path.append('/home/dennis/git/bootstrapcb/python')
import bootstrapcb as cb
#########################################################################

# get chart properties
props = chart.get_properties()
utils.preconfigure_plot(props)

# collect parameters for query 
filter_expression = props["filter"]
xaxis_itervar = props["xaxis_itervar"]
iso_itervar = props["iso_itervar"] 

# query data into a data frame
df = results.get_scalars(filter_expression, include_runattrs=True, include_attrs=True, include_itervars=True)

if df.empty:
    plot.set_warning("The result filter returned no data.")
    exit(1)

if not xaxis_itervar and not iso_itervar:
    print("The X Axis and Iso Line options were not set in the dialog, inferring them from the data..")
    xaxis_itervar, iso_itervar = utils.pick_two_columns(df)
    if not xaxis_itervar:
        plot.set_warning("Please set the X Axis (and Iso Lines) options in the dialog!")
        exit(1)

utils.assert_columns_exist(df, [xaxis_itervar])
df[xaxis_itervar] = pd.to_numeric(df[xaxis_itervar], errors="ignore")

if iso_itervar:
    utils.assert_columns_exist(df, [xaxis_itervar])
    df[iso_itervar] = pd.to_numeric(df[iso_itervar], errors="ignore")

if not iso_itervar: # might be an empty string
    iso_itervar = None

if df.empty:
    plot.set_warning("Select scalars for the Y axis in the Properties dialog")
    exit(1)

unique_names = df["name"].unique()

if len(unique_names) != 1:
    plot.set_warning("Selected scalars must share the same name.")
    exit(1)

scalar_name = unique_names[0]

confidence_level_str = props["confidence_level"] if "confidence_level" in props else "none"

if confidence_level_str == "none":
    df = pd.pivot_table(df, values="value", columns=iso_itervar, index=xaxis_itervar)
else:
    confidence_level = float(confidence_level_str[:-1])
    method_str = props["method"]
    p0 = eval(props["p0"])
    f_str = props["model_func"]
    pdf_str = props["pdf"] if "pdf" in props else "none"
    cdf_str = props["cdf"] if "cdf" in props else "none"
    if f_str: 
        code = f_str.split("\n")
        parsed_code = "def f(x,theta):\n"
        for line in code: parsed_code += "  "+line+"\n"
        exec(parsed_code)
    if pdf_str: 
        code = pdf_str.split("\n")
        parsed_code = "def pdf(x,y,theta):\n"
        for line in code: parsed_code += "  "+line+"\n"
        exec(parsed_code)
    if cdf_str: 
        code = cdf_str.split("\n")
        parsed_code = "def cdf(x,y,theta):\n"
        for line in code: parsed_code += "  "+line+"\n"
        exec(parsed_code)
    #xs = df[xaxis_itervar].values
    #ys = df["value"].values
    #x = np.linspace(xs[0],xs[-1],num=100)
#########################################################################
    xs = [0]*47 
    ys = [  4.3, 4.7, 4.7, 3.1, 5.2, 6.7, 4.5, 3.6, 7.2, 10.9,  6.6, 5.8,
            6.3, 4.7, 8.2, 6.2, 4.2, 4.1, 3.3, 4.6, 6.3,  4.0,  3.1, 3.5,
            7.8, 5.0, 5.7, 5.8, 6.4, 5.2, 8.0, 4.9, 6.1,  8.0,  7.7, 4.3,
           12.5, 7.9, 3.9, 4.0, 4.4, 6.7, 3.8, 6.4, 7.2,  4.8, 10.5       ]
    x = np.linspace(0,0.1,num=100)
#########################################################################
    opt, V, I = cb.mle(xs, ys, pdf, p0, info=True) 
    mean = [f(x_i, opt) for x_i in x] 
    df = pd.DataFrame(data={ 
        "x": x, 
        "mean": mean  
    }) 
    df = df.set_index(["x"])  
    if method_str == "Pointwise delta-method":
        conf_l, conf_u = cb.conf_band_delta(x, confidence_level, f, opt, V)
    elif method_str == "Simultaneous delta-method":
        conf_l, conf_u = cb.conf_band_delta(x, confidence_level, f, opt, V)
    elif method_str == "Simultaneous delta-method (bootstrapped)":
        conf_l, conf_u = cb.conf_band_delta(x, confidence_level, f, opt, V)
    elif method_str == "Bootstrap likelihood-based region R_alpha":
        conf_l, conf_u = cb.conf_band_delta(x, confidence_level, f, opt, V)
    elif method_str == "Bootstrap likelihood-based region dR_alpha":
        conf_l, conf_u = cb.conf_band_delta(x, confidence_level, f, opt, V)
    elif method_str == "Bootstrap likelihood-based region dR_alpha (nelder-mead)":
        conf_l, conf_u = cb.conf_band_delta(x, confidence_level, f, opt, V)
    df["cb_l"] = conf_l 
    df["cb_u"] = conf_u       

legend_cols, _ = utils.extract_label_columns(df)

p = plot if chart.is_native_chart() else plt


p.xlabel(xaxis_itervar)
p.ylabel(scalar_name)

try:
    xs = pd.to_numeric(df.index.values)
except:
    plot.set_warning("The X axis itervar is not purely numeric, this is not supported yet.")
    exit(1)

for c in df.columns:
    style = utils._make_line_args(props, c, df)
    ys = df[c].values
    p.plot(xs, ys, label=(iso_itervar + "=" + str(df[c].name) if iso_itervar else scalar_name), **style)

utils.set_plot_title(scalar_name + " vs. " + xaxis_itervar)

utils.postconfigure_plot(props)

utils.export_image_if_needed(props)
utils.export_data_if_needed(df, props)
