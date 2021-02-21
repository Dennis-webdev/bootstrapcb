import math
import numpy as np
import numdifftools as nd
import pandas as pd
import matplotlib.pyplot as plt
from omnetpp.scave import results, chart, utils, plot, vectorops as ops
from scipy.integrate import quad

#########################################################################
import sys
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

# get data
#########################################################################
xdata = [2.0, 7.0, 12.0, 19.5, 29.5, 39.5, 54.5, 75.0]
ydata = [ [ 1.26,   2.78,  0.63,  0.34], 
          [ 3.53,    4.1,  1.31,  0.91],
          [11.98,  13.14,  9.86,  6.53], 
          [90.82,  97.12, 75.85, 59.46], 
          [83.45, 116.62,  104., 80.85], 
          [55.98,  67.28, 79.33, 82.66],
          [66.32,  78.53,  69.1, 67.27], 
          [39.42,  55.35, 60.76,  73.2] ]
xs, ys = [], []
for i in range(len(xdata)):
    for j in range(len(ydata[i])):
        xs.append(xdata[i])
        ys.append(ydata[i][j])
#########################################################################
#xs = df[xaxis_itervar].values
#ys = df["value"].values
x = np.linspace(xs[0],xs[-1],num=100)
    
f_str = props["model_func"]
if f_str:
    # parse f
    code = f_str.split("\n")
    parsed_code = "def f(x,theta):\n"
    for line in code: parsed_code += "  "+line+"\n"
    exec(parsed_code)
    # parse p0
    p0 = None
    p0_str = props["p0"]
    try:  
        p0 = eval(p0_str)
    except: 
        plot.set_warning("Parameter not given.")
        exit(1)
    # get pdf if given
    pdf = None
    pdf_str = props["pdf"]
    if pdf_str: 
        code = pdf_str.split("\n")
        parsed_code = "def pdf(x,y,theta):\n"
        for line in code: parsed_code += "  "+line+"\n"
        exec(parsed_code) 
    # get cdf if given
    cdf = None
    cdf_str = props["cdf"] 
    if cdf_str: 
        code = cdf_str.split("\n")
        parsed_code = "def cdf(x,y,theta):\n"
        for line in code: parsed_code += "  "+line+"\n"
        exec(parsed_code)  
    # get class instance
    bscb = cb.cb_class(xdata, ydata, f, p0, pdf, cdf)  
    # MLE
    mean, V = bscb.mle(x)   
    df = pd.DataFrame(data={ 
        "x": x, 
        "mean": mean  
    }) 
    df = df.set_index(["x"])
 
confidence_level_str = props["confidence_level"] 
if f_str and confidence_level_str:   
    method_str = props["method"]           
    # calculate CB
    confidence_level = float(confidence_level_str[:-1])
    c = 0
    if method_str == "Pointwise delta-method":
        conf_l, conf_u = bscb.conf_band_delta(x, confidence_level)
        df["cb_l"+"*"*c] = conf_l   
        df["cb_u"+"*"*c] = conf_u
        c += 1
    if method_str == "Simultaneous delta-method":
        conf_l, conf_u = bscb.conf_band_delta(x, confidence_level)
        df["cb_l"+"*"*c] = conf_l   
        df["cb_u"+"*"*c] = conf_u
        c += 1
    if method_str == "Simultaneous delta-method (bootstrapped)":
        conf_l, conf_u = bscb.conf_band_delta(x, confidence_level)
        df["cb_l"+"*"*c] = conf_l   
        df["cb_u"+"*"*c] = conf_u
        c += 1 
    if method_str == "Bootstrap likelihood-based region R_alpha":
        conf_l, conf_u = bscb.conf_band_bs_ralpha(x, confidence_level) 
        df["cb_l"+"*"*c] = conf_l    
        df["cb_u"+"*"*c] = conf_u
        c += 1
    if method_str == "Bootstrap likelihood-based region dR_alpha":
        conf_l, conf_u = bscb.conf_band_bs_dralpha(x, confidence_level)
        df["cb_l"+"*"*c] = conf_l   
        df["cb_u"+"*"*c] = conf_u 
        c += 1 
    if method_str == "Bootstrap likelihood-based region dR_alpha (nelder-mead)":
        conf_l, conf_u = bscb.conf_band_approx_dralpha(x, confidence_level)
        df["cb_l"+"*"*c] = conf_l   
        df["cb_u"+"*"*c] = conf_u   
        c += 1     

legend_cols, _ = utils.extract_label_columns(df) 

p = plot if chart.is_native_chart() else plt

 
p.xlabel(xaxis_itervar)
p.ylabel(scalar_name)

# plot data
p.plot(xs,ys,label='data',linestyle='',marker='o',color='black',markersize='3')

# plot regression
if f_str:
    try:
        xs = pd.to_numeric(df.index.values)
    except:
        plot.set_warning("The X axis itervar is not purely numeric, this is not supported yet.")
        exit(1)
    
    for c in df.columns:
        style = utils._make_line_args(props, c, df)
        ys = df[c].values
        p.plot(xs, ys, label=df[c].name, **style)

utils.set_plot_title(scalar_name + " vs. " + xaxis_itervar)

utils.postconfigure_plot(props)

utils.export_image_if_needed(props)
utils.export_data_if_needed(df, props)
