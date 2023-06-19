import datetime
import pandas as pd
from ydata_profiling import ProfileReport
from scipy.stats import boxcox

csv = pd.read_csv(r"ppl_burstiness_preprocessed.csv")

transformed_perplexity, _ = boxcox(csv['perplexity'])
transformed_burstiness, _ = boxcox(csv['burstiness'])

csv['perplexity'] = transformed_perplexity
csv['burstiness'] = transformed_burstiness

# I need to split data into training and testing data, then transform them separately using the same methods *********

# csv.to_csv('data_boxcox.csv')
#
# profile = ProfileReport(csv)
#
# profile.to_file(output_file="data_report_boxcox.html")

print("Program finished executing at:", datetime.datetime.now())
