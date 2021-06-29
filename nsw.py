import os
from datetime import datetime
from pytz import timezone
from pathlib import Path

from scipy.optimize import curve_fit
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd

converter = mdates.ConciseDateConverter()

munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter


# Data from covidlive by date announced to public
def covidlive_data(start_date=np.datetime64('2021-06-10')):
    df = pd.read_html('https://covidlive.com.au/report/daily-source-overseas/nsw')[1]

    df = df[:200]

    if df['NET'][0] == '-':
        df = df[1:200]

    dates = np.array(
        [
            np.datetime64(datetime.strptime(date, "%d %b %y"), 'D') - 1
            for date in df['DATE']
        ]
    )
    cases = np.array(df['NET'].astype(int))
    cases = cases[dates >= start_date][::-1]
    dates = dates[dates >= start_date][::-1]

    return dates, cases


# Data from NSW Health by test notification date
def nswhealth_data(start_date=np.datetime64('2021-06-10')):
    url = (
        "https://data.nsw.gov.au/data/dataset/"
        "c647a815-5eb7-4df6-8c88-f9c537a4f21e/"
        "resource/2f1ba0f3-8c21-4a86-acaf-444be4401a6d/"
        "download/confirmed_cases_table3_likely_source.csv"
    )
    df = pd.read_csv(url)

    LOCAL = [
        'Locally acquired - no links to known case or cluster',
        'Locally acquired - investigation ongoing',
        'Locally acquired - linked to known case or cluster',
    ]

    cases_by_date = {
        d: 0
        for d in np.arange(
            np.datetime64(df['notification_date'].min()),
            np.datetime64(df['notification_date'].max()) + 1,
        )
    }

    for _, row in df.iterrows():
        if row['likely_source_of_infection'] in LOCAL:
            cases_by_date[np.datetime64(row['notification_date'])] += 1


    dates = np.array(list(cases_by_date.keys()))
    new = np.array(list(cases_by_date.values()))

    return dates[dates >= start_date], new[dates >= start_date]


def nonisolating_data():
    DATA = """
        2021-06-10 0
        2021-06-11 0
        2021-06-12 0
        2021-06-13 0
        2021-06-14 0
        2021-06-15 0
        2021-06-16 0
        2021-06-17 4
        2021-06-18 1
        2021-06-19 2
        2021-06-20 1
        2021-06-21 0
        2021-06-22 3
        2021-06-23 12
        2021-06-24 4
        2021-06-25 9
        2021-06-26 12
        2021-06-27 19
        2021-06-28 13
        2021-06-29 12
    """

    def unpack_data(s):
        dates = []
        values = []
        for line in s.splitlines():
            if line.strip() and not line.strip().startswith('#'):
                date, value = line.strip().split(maxsplit=1)
                dates.append(np.datetime64(date) - 1)
                values.append(eval(value))
        return np.array(dates), np.array(values)

    dates, new = unpack_data(DATA)
    return dates, new


def gaussian_smoothing(data, pts):
    """gaussian smooth an array by given number of points"""
    x = np.arange(-4 * pts, 4 * pts + 1, 1)
    kernel = np.exp(-(x ** 2) / (2 * pts ** 2))
    smoothed = convolve(data, kernel, mode='same')
    normalisation = convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / normalisation


def fourteen_day_average(data):
    ret = np.cumsum(data, dtype=float)
    ret[14:] = ret[14:] - ret[:-14]
    return ret / 14


def partial_derivatives(function, x, params, u_params):
    model_at_center = function(x, *params)
    partial_derivatives = []
    for i, (param, u_param) in enumerate(zip(params, u_params)):
        d_param = u_param / 1e6
        params_with_partial_differential = np.zeros(len(params))
        params_with_partial_differential[:] = params[:]
        params_with_partial_differential[i] = param + d_param
        model_at_partial_differential = function(x, *params_with_partial_differential)
        partial_derivative = (model_at_partial_differential - model_at_center) / d_param
        partial_derivatives.append(partial_derivative)
    return partial_derivatives


def model_uncertainty(function, x, params, covariance):
    u_params = [np.sqrt(abs(covariance[i, i])) for i in range(len(params))]
    derivs = partial_derivatives(function, x, params, u_params)
    squared_model_uncertainty = sum(
        derivs[i] * derivs[j] * covariance[i, j]
        for i in range(len(params))
        for j in range(len(params))
    )
    return np.sqrt(squared_model_uncertainty)


dates, new = nswhealth_data()

for d, n in zip(dates, new):
    print(d, n)

# If NSW health data not updated yet, use covidlive data:
# cl_dates, cl_new = covidlive_data(start_date=dates[-1] + 1)
# dates = np.append(dates, cl_dates)
# new = np.append(new, cl_new)

dates, new = covidlive_data()
# dates, new = nonisolating_data()
# for d, n in zip(dates, new):
#     print(d, n)

# dates = np.append(dates, [dates[-1] + 1])
# new = np.append(new, [28])

START_PLOT = start_date=np.datetime64('2021-06-13')
END_PLOT = np.datetime64('2021-08-01')

SMOOTHING = 4
PADDING = 3 * int(round(3 * SMOOTHING))
new_padded = np.zeros(len(new) + PADDING)
new_padded[: -PADDING] = new


def exponential(x, A, k):
    return A * np.exp(k * x)


# def linear(x, A, B):
#     return A * x + B


tau = 5  # reproductive time of the virus in days

# Smoothing requires padding to give sensible results at the right edge. Compute an
# exponential fit to daily cases over the last fortnight, and pad the data with the
# fit results prior to smoothing.

FIT_PTS = min(20, len(dates[dates >= START_PLOT]))
x0 = -7
delta_x = 1
fit_x = np.arange(-FIT_PTS, 0)
fit_weights = 1 / (1 + np.exp(-(fit_x - x0) / delta_x))
pad_x = np.arange(PADDING)

def clip_params(params):
    # Clip exponential fit params to be within a reasonable range to suppress when
    # unlucky points lead us to an unrealistic exponential blowup. Modifies array
    # in-place.
    R_CLIP = 5 # Limit the exponential fits to a maximum of R=5
    params[0] = min(params[0], 2 * new[-FIT_PTS:].max() + 1)
    params[1] = min(params[1], np.log(R_CLIP ** (1 / tau)))


params, cov = curve_fit(exponential, fit_x, new[-FIT_PTS:], sigma=1 / fit_weights)
clip_params(params)
fit = exponential(pad_x, *params).clip(0.1, None)

# Linear fit for now
# params, cov = curve_fit(linear, fit_x, new[-FIT_PTS:], sigma=1 / fit_weights)
# fit = linear(pad_x, *params).clip(0.1, None)


new_padded[-PADDING:] = fit
new_smoothed = gaussian_smoothing(new_padded, SMOOTHING)[: -PADDING]
# new_smoothed = gaussian_smoothing(new, SMOOTHING)
R = (new_smoothed[1:] / new_smoothed[:-1]) ** tau

# def correct_smoothing(new_smoothed, R):
#     # Gaussian smoothing creates a consistent bias whenever there is curvature. Measure
#     # and correct for it
#     f = R ** (SMOOTHING / 5)
#     bias =  (new_smoothed[1:] * f - new_smoothed[1:] / f) / 2 - new_smoothed[1:]
#     new_smoothed[1:] -= bias
#     new_smoothed[0] -= bias[0]
#     return new_smoothed

# new_smoothed = correct_smoothing(new_smoothed, R)

N_monte_carlo = 1000
variance_R = np.zeros_like(R)
variance_new_smoothed = np.zeros_like(new_smoothed)
cov_R_new_smoothed = np.zeros_like(R)
# Monte-carlo of the above with noise to compute variance in R, new_smoothed,
# and their covariance:
u_new = np.sqrt((0.2 * new) ** 2 + new)  # sqrt(N) and 20%, added in quadrature
for i in range(N_monte_carlo):
    new_with_noise = np.random.normal(new, u_new).clip(0.1, None)
    params, cov = curve_fit(
        exponential,
        fit_x,
        new_with_noise[-FIT_PTS:],
        sigma=1 / fit_weights,
        maxfev=20000,
    )
    clip_params(params)
    scenario_params = np.random.multivariate_normal(params, cov)
    clip_params(scenario_params)
    fit = exponential(pad_x, *scenario_params).clip(0.1, None)

    # Linear for now:
    # params, cov = curve_fit(
    #     linear,
    #     fit_x,
    #     new_with_noise[-FIT_PTS:],
    #     sigma=1 / fit_weights,
    #     maxfev=20000,
    # )
    # scenario_params = np.random.multivariate_normal(params, cov)
    # fit = linear(pad_x, *scenario_params).clip(0.1, None)


    new_padded[:-PADDING] = new_with_noise
    new_padded[-PADDING:] = fit
    new_smoothed_noisy = gaussian_smoothing(new_padded, SMOOTHING)[:-PADDING]
    # new_smoothed_noisy = gaussian_smoothing(new_with_noise, SMOOTHING)
    variance_new_smoothed += (new_smoothed_noisy - new_smoothed) ** 2 / N_monte_carlo
    R_noisy = (new_smoothed_noisy[1:] / new_smoothed_noisy[:-1]) ** tau
    variance_R += (R_noisy - R) ** 2 / N_monte_carlo
    cov_R_new_smoothed += (
        (new_smoothed_noisy[1:] - new_smoothed[1:]) * (R_noisy - R) / N_monte_carlo
    )

u_R = np.sqrt(variance_R)
R_upper = R + u_R
R_lower = R - u_R

u_new_smoothed = np.sqrt(variance_new_smoothed)
new_smoothed_upper = new_smoothed + u_new_smoothed
new_smoothed_lower = new_smoothed - u_new_smoothed

R_upper = R_upper.clip(0, 10)
R_lower = R_lower.clip(0, 10)
R = R.clip(0, None)

new_smoothed_upper = new_smoothed_upper.clip(0, None)
new_smoothed_lower = new_smoothed_lower.clip(0, None)
new_smoothed = new_smoothed.clip(0, None)

# Propagate uncertainty in log space where linear uncertainty propagation better
# applies
def log_projection_model(t, A, R):
    return np.log(A * R ** (t / tau))

# Projection of daily case numbers:
days_projection = (END_PLOT - dates[-1]).astype(int)
t_projection = np.linspace(0, days_projection, days_projection + 1)

# Construct a covariance matrix for the latest estimate in new_smoothed and R:
cov = np.array(
    [
        [variance_new_smoothed[-1], cov_R_new_smoothed[-1]],
        [cov_R_new_smoothed[-1], variance_R[-1]],
    ]
)

new_projection = np.exp(log_projection_model(t_projection, new_smoothed[-1], R[-1]))
log_new_projection_uncertainty = model_uncertainty(
    log_projection_model, t_projection, (new_smoothed[-1], R[-1]), cov
)
new_projection_upper = np.exp(np.log(new_projection) + log_new_projection_uncertainty)
new_projection_lower = np.exp(np.log(new_projection) - log_new_projection_uncertainty)

# Examining whether the smoothing and uncertainty look decent
# plt.bar(dates, new)
# plt.fill_between(
#     dates,
#     new_smoothed_lower,
#     new_smoothed_upper,
#     color='orange',
#     alpha=0.5,
#     zorder=5,
#     linewidth=0,
# )
# plt.plot(dates, new_smoothed, color='orange', zorder=6)
# plt.plot(
#     dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
#     new_projection,
#     color='orange',
#     zorder=6,
# )
# plt.fill_between(
#     dates[-1] + 24 * t_projection.astype('timedelta64[h]'),
#     new_projection_lower,
#     new_projection_upper,
#     color='orange',
#     alpha=0.5,
#     zorder=5,
#     linewidth=0,
# )
# params, cov = curve_fit(exponential, fit_x, new[-FIT_PTS:], sigma=1 / fit_weights)
# clip_params(params)
# fit = exponential(fit_x, *params).clip(0.1, None)

# plt.plot(dates[-1] + 1 + fit_x, fit)
# plt.grid(True)
# plt.axis(xmin=dates[0], xmax=dates[-1] + 14, ymin=0, ymax=2 * new[-1])
# plt.show()

MASKS = np.datetime64('2021-06-21')
LGA_LOCKDOWN = np.datetime64('2021-06-26')
LOCKDOWN = np.datetime64('2021-06-27')

ORANGEYELLOW = (
    np.array(mcolors.to_rgb("orange")) + np.array(mcolors.to_rgb("yellow"))
) / 2

fig1 = plt.figure(figsize=(10, 6))
plt.fill_betweenx(
    [-10, 10],
    [MASKS, MASKS],
    [LGA_LOCKDOWN, LGA_LOCKDOWN],
    color="yellow",
    alpha=0.3,
    linewidth=0,
    label="Initial restrictions",
)
plt.fill_betweenx(
    [-10, 10],
    [LGA_LOCKDOWN, LGA_LOCKDOWN],
    [LOCKDOWN, LOCKDOWN],
    color="orange",
    alpha=0.3,
    linewidth=0,
    label="East Sydney LGA lockdown",
)
plt.fill_betweenx(
    [-10, 10],
    [LOCKDOWN, LOCKDOWN],
    [LOCKDOWN + 14, LOCKDOWN + 14],
    color="red",
    alpha=0.3,
    linewidth=0,
    label="Greater Sydney lockdown",
)

for i in range(10):
    plt.fill_betweenx(
        [-10, 10],
        [LOCKDOWN.astype(int) + 14 + 0.5 * i, LOCKDOWN.astype(int) + 14 + 0.5 * i],
        [
            LOCKDOWN.astype(int) + 14 + 0.5 * i + 0.5,
            LOCKDOWN.astype(int) + 14 + 0.5 * i + 0.5,
        ],
        color="red",
        alpha=0.3 * (10 - i) / 10,
        linewidth=0,
        zorder=-10,
    )



plt.fill_between(
    dates[1:] + 1,
    R,
    label=R"$R_\mathrm{eff}$",
    step='pre',
    color='C0',
)

plt.fill_between(
    dates[1:] + 1,
    R_lower,
    R_upper,
    label=R"$R_\mathrm{eff}$ uncertainty",
    color='cyan',
    edgecolor='blue',
    alpha=0.2,
    step='pre',
    zorder=2,
    # linewidth=0,
    hatch="////",
)

plt.axhline(1.0, color='k', linewidth=1)
plt.axis(xmin=START_PLOT, xmax=END_PLOT, ymin=0, ymax=6)
plt.grid(True, linestyle=":", color='k', alpha=0.5)

handles, labels = plt.gca().get_legend_handles_labels()

plt.ylabel(R"$R_\mathrm{eff}$")

u_R_latest = (R_upper[-1] - R_lower[-1]) / 2

plt.title(
    "$R_\\mathrm{eff}$ in New South Wales with Sydney restriction levels and daily cases"
    + (
        "\n"
        + fR"Latest estimate: $R_\mathrm{{eff}}={R[-1]:.01f} \pm {u_R_latest:.01f}$"
    )
)

plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax2 = plt.twinx()
plt.step(dates + 1, new + 0.02, color='purple', label='Daily cases')
plt.semilogy(
    dates.astype(int) + 0.5,
    new_smoothed,
    color='magenta',
    label='Daily cases (smoothed)',
)

plt.fill_between(
    dates.astype(int) + 0.5,
    new_smoothed_lower,
    new_smoothed_upper,
    color='magenta',
    alpha=0.3,
    linewidth=0,
    zorder=10,
    label='Smoothing/trend uncertainty',
)
plt.plot(
    dates[-1].astype(int) + 0.5 + t_projection,
    new_projection,
    color='magenta',
    linestyle='--',
    label='Daily cases (trend)',
)
plt.fill_between(
    dates[-1].astype(int) + 0.5 + t_projection,
    new_projection_lower,
    new_projection_upper,
    color='magenta',
    alpha=0.3,
    linewidth=0,
)
plt.axis(ymin=1, ymax=1000)
plt.ylabel("Daily confirmed cases")
plt.tight_layout()

handles2, labels2 = plt.gca().get_legend_handles_labels()

handles += handles2
labels += labels2

order = [3, 4, 5, 6, 7, 8, 0, 1, 2]
plt.legend(
    # handles,
    # labels,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='upper right',
    ncol=1,
)

plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
plt.gca().tick_params(axis='y', which='minor', labelsize='x-small')
plt.setp(plt.gca().get_yminorticklabels()[1::2], visible=False)
plt.gca().xaxis.set_major_locator(mdates.DayLocator([1, 5, 10, 15, 20, 25]))
plt.gca().get_xaxis().get_major_formatter().show_offset = False

fig1.savefig('COVID_NSW.svg')
fig1.savefig('COVID_NSW.png', dpi=600)
plt.show()

# Update the date in the HTML
html_file = 'COVID_NSW.html'
html_lines = Path(html_file).read_text().splitlines()
now = datetime.now(timezone('Australia/Melbourne')).strftime('%Y-%m-%d-%H:%M')
for i, line in enumerate(html_lines):
    if 'Last updated' in line:
        html_lines[i] = f'    Last updated: {now} AEST'
Path(html_file).write_text('\n'.join(html_lines) + '\n')
plt.show()
