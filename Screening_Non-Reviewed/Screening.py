import scipy
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData

ROOT_PATH = os.path.join("C:\\Users\\alexa\\OneDrive\\Documents\\Uni\\Code\\PhytonScripts")
SCREENING_PATH = os.path.join("I:\\ScreeningSimulations")

sys.path.append(os.path.join(ROOT_PATH, "Plotting"))
sys.path.append(os.path.join(ROOT_PATH, "PulseSimulation"))
from TexToUni import tex_to_uni
import PlottingTool as ptool
import ReadSimulation as rs

THIS_PATH = os.path.join(ROOT_PATH, "Screening")

def get_energy(channels, cal_channels, cal_energies):
    popt, _ = curve_fit(
        lambda x, a0, a1, a2: a0 + a1 * x + a2 * x**2,
        cal_channels, cal_energies, p0=[10, 1e-1, 1e-6])
    return popt[0] + popt[1] * channels + popt[2] * channels**2


def gauss_fit(x, a, dx, s, c0, c1):
    return np.abs(a) * np.exp(- (x - dx)**2 / (2. * s**2)) + c0 + c1 * x


def get_area(popt):
    out = np.abs(popt[0]) * np.sqrt(2. * np.pi) * \
            scipy.special.erf(5. / np.sqrt(2.)) * np.abs(popt[2])
    return out


def get_area_error(popt, cov):
    c = np.sqrt(2. * np.pi) * scipy.special.erf(5. / np.sqrt(2.))
    e = popt[0]**2 * np.diag(cov)[2] + popt[2]**2 * np.diag(cov)[0]
    return c * np.sqrt(e)


def get_dec(number):
    if (number == 0):
        return 0
    dec = 0
    t_num = number
    while (t_num < 1):
        dec += 1
        t_num *= 10
    return dec


def get_string(number, dec=0):
    out_number = str(number)
    out_str = out_number.split('.')[0]
    if (len(out_number.split('.')) > 1):
        tmp = out_number.split('.')[1]
        _d = 0
        out_str += '.'
        while (_d < dec) and (len(tmp) > 2):
            out_str += tmp[0]
            tmp = tmp[1:]
            _d += 1

        index = -1
        t0 = int(tmp[0])
        while (t0 >= 5):
            if out_str[index] == ".":
                index -= 1
                continue
            t1 = int(out_str[index])
            if (t1 + 1 == 10):
                if index < -1:
                    out_str = out_str[:index] + "0" + out_str[index + 1:]
                else:
                    out_str = out_str[:index] + "0"
                index -= 1
            else:
                if index < -1:
                    out_str = out_str[:index] + str(t1 + 1) + out_str[index + 1:]
                else:
                    out_str = out_str[:index] + str(t1 + 1)
                break

        if (dec == 0):
            out_str = out_str[:-1]

    return out_str

XLABEL = "Energy in keV"
DATA_PATH = os.path.join(THIS_PATH, "ECHo_Screening")
LENGTH = 16382

DATA_LIST = np.array([
    "20150820_Background.TKA",
    "20150918_Alu.TKA",
    "20151015_Copper.TKA",
    "20161229_Background.TKA",
    "20170320_Alu.TKA",
    "20170607_Connectors.TKA",
    "20170612_PCB.TKA"])

TITLE_LIST = np.array([
    "Background 20.08.15: ",        # 0
    "Cryoperm 18.09.15: ",               # 1
    "Copper 15.10.15: ",            # 2
    "Background 29.12.16: ",        # 3
    "Alu 20.03.17: ",               # 4
    "Connectors 07.06.17: ",        # 5
    "Circuit board 12.06.17: "])    # 6

CAL_E = np.array([239, 511, 1173, 1332, 1461, 2615])
CAL_C = np.array([np.array([1198, 2598, 5993, 6810, 7465, 13373]),  # 0
                  np.array([1201, 2604, 6005, 6824, 7484, 13404]),  # 1
                  np.array([1201, 2604, 6005, 6824, 7484, 13404]),  # 2
                  np.array([1210, 2623, 6047, 6870, 7535, 13495]),  # 3
                  np.array([1210, 2624, 6050, 6873, 7537, 13502]),  # 4
                  np.array([1210, 2624, 6050, 6873, 7537, 13502]),  # 5
                  np.array([1210, 2624, 6050, 6873, 7537, 13502])]) # 6

def load_data(sample, bulk=True):
    ENERGY_AXIS = "Deposited Energy in meV"
    _type = "Surface"
    l = [1., 10., 14.]
    if bulk:
        _type = "Bulk"
        l = [1., 10., 14.]
    types = np.array(["K-40", "Th-232", "U-238"])
    out = {}
    norm = {}
    for i, t in enumerate(types):
        dir_name = t + "_" + _type
        path = os.path.join(SCREENING_PATH, sample, dir_name)
        pathes = []
        seeds = []
        out[t] = None
        num_of_events = 0

        for dataset in os.scandir(path):
            if os.path.isdir(dataset.path):
                skip_this = False

                tmp = pd.read_csv(os.path.join(
                    dataset.path, dir_name + "_Source.csv"))
                tmp_seed = [tmp["Seed X"].iloc[0], tmp["Seed Y"].iloc[0]]

                for f in os.scandir(dataset.path):
                    if f.name[-11:] == "_Events.txt":
                        num_of_events += int(f.name.split("_Events.txt")[0])
                    elif os.path.isdir(f.path) and (f.name == "Event"):
                        if len(os.listdir(f.path)) == 0:
                            skip_this = True

                if tmp_seed not in seeds:
                    seeds.append(tmp_seed)
                    if not skip_this:
                        pathes.append(dataset.name)
                else:
                    print("Warning: Equal seeds occure -- " + dataset.name)

        for p in pathes:
            tmp = rs.Simulation(os.path.join(path, p), primaries=False, events=True)
            tmp_e = tmp.Events.Table
            if False:#(_type == "Surface") and (t == "U-238"):
                tmp_e = tmp_e[tmp_e["Origin"] % 1e4 >= 2230]
                for nuc in np.unique(tmp_e["Origin"]):
                    _scale_m = range(1)
                    tmp_data = tmp_e[tmp_e["Origin"] == nuc][ENERGY_AXIS]

                    if nuc % 1e4 > 40:
                        _scale_m = range(7)

                    for _i in _scale_m:
                        if out[t] is None:
                            out[t] = tmp_data.copy()
                        else:
                            out[t] = np.append(out[t], tmp_data)
            else:
                if out[t] is None:
                    out[t] = tmp_e[ENERGY_AXIS].copy()
                else:
                    out[t] = np.append(out[t], tmp_e[ENERGY_AXIS])
        out[t] = out[t] * 1e-6
        norm[t] = num_of_events * l[i]
    return out, norm



def plot_screening(plot_index, plot_bg=False, sim_path=None,
        cal_info=False, peak_info=False, bulk=True):

    data_path = os.path.join(DATA_PATH, DATA_LIST[plot_index])
    data = np.loadtxt(data_path)[2:]
    norm = np.loadtxt(data_path)[0]

    _index = plot_index

    mass = 1
    area = 1

    if sim_path is not None:
        if "PCB" in sim_path:
            mass = 0.014
            area = 153
        elif "Copper" in sim_path:
            mass = 0.026
            area = 23
        elif "Cryoperm" in sim_path:
            mass = 0.142
            area = 327
        elif "Connector" in sim_path:
            mass = 0.0075
            area = 55.5


    if plot_index < 3:
        bg_index = 0
        cal_c = np.array([1201, 2604, 6005, 6824, 7484, 13404])
    else:
        bg_index = 3
        cal_c = np.array([1210, 2624, 6050, 6873, 7537, 13502])

    background_path = os.path.join(DATA_PATH, DATA_LIST[bg_index])
    bg_data = np.loadtxt(background_path)[2:]
    bg_norm = np.loadtxt(background_path)[0]

    if plot_bg:
        _index = bg_index
        data = bg_data
        norm = bg_norm

    title = TITLE_LIST[_index]
    xdata = get_energy(np.arange(LENGTH), cal_c, CAL_E)
    ENERGY_PER_BIN = xdata[1] - xdata[0]
    if cal_info:
        print("Bins: " + str(LENGTH))
        print("Min energy: " + str(xdata[0]) + " keV")
        print("Max energy: " + str(xdata[-1]) + " keV")
    YLABEL = tex_to_uni("10^{-3}\,counts\,s^{-1}\,(" +
                        str(int(1e3 * ENERGY_PER_BIN)) + "\,eV)^{-1}")

    #xdata = get_energy(np.arange(16382), **background_easter_cal)
    #print(xdata[1] - xdata[0])

    DURATION = str(norm / (3600 * 24))
    D0 = DURATION.split('.')[0]
    D1 = DURATION.split('.')[1]

    d_app = D1[0]
    if (len(D1) > 1) and (int(D1[1]) >= 5):
        d_app = str(int(d_app) + 1)
    d_app = "Duration = " + D0 + "." + d_app + " days"

    if cal_info:
        cal_out = ptool.Curve(
            np.arange(LENGTH), data, title=title + d_app,
            log=True, color=0, xlabel="Channel", ylabel=YLABEL,
            norm=1e-3*norm, legendpos=False)
        cal_out.plot()

    out = ptool.Curve(
        xdata, data, title=title + d_app,
        log=True, color=0, xlabel=XLABEL, ylabel=YLABEL,
        norm=1e-3*norm, legendpos=False)

    def delta(data, mod=0):
        if mod == 0:
            fe = 13400
            fit_y = data[fe:13600]   # Tl-208 2615 keV
        elif mod == 1:
            fe = 7450
            fit_y = data[fe:7600]      # K-40 1461 keV
        else:
            fe = 1760
            fit_y = data[fe:1810]        # Pb-214 352 keV

        fit_x = (np.arange(len(fit_y)) + fe) * ENERGY_PER_BIN

        gauss_p0 = [fit_y.max() / 2., (np.argmax(fit_y) + fe) * ENERGY_PER_BIN, 5e-1, 5e-5, -1e-8]
        bounds = [[0., 0., 0.1, -np.inf, -np.inf], [np.inf, 3000, 10, np.inf, 0]]
        popt, cov = curve_fit(gauss_fit, fit_x, fit_y, p0=gauss_p0, bounds=bounds)

        return popt[1]# - fe

    tl_pos = delta(data)
    k_pos = delta(data, mod=1)
    pb_pos = delta(data, mod=2)

    if sim_path is not None:
        s_path = os.path.join(THIS_PATH, "ScreeningEvaluation", sim_path)

        if bulk:
            sim_names = ["K-40_Bulk", "Th-232_Bulk", "U-238_Bulk"]
        else:
            sim_names = ["K-40_Surface", "Th-232_Surface", "U-238_Surface"]

        SIM_L = len(sim_names)
        tmp_i = SIM_L

        sim_data = {}
        _sim_data, _sim_norm = load_data(sim_path, bulk=bulk)
        e_res = 2e3 / (2.3548 * 1e3 * ENERGY_PER_BIN)
        #print("Xaxis")
        #print(xdata)
        #print(len(xdata))

        energies = {}
        energies2 = {}
        for name in sim_names:
            energies[name] = _sim_data[name.split("_")[0]]
            energies2[name] = energies[name]**2

        min_e = xdata[0]
        max_e = xdata[-1] + ENERGY_PER_BIN

        def get_scale(B, x):
            out = np.zeros(3)
            a = (B[0] + 10000) * 1e-4
            b = B[1] * 1e-7
            c = B[2] * 1e-1

            for i, name in enumerate(sim_names):
                values, edges = np.histogram(
                    a * energies[name] - b * energies2[name] + c,
                    bins=LENGTH, range=(min_e, max_e))
                spect = scipy.ndimage.gaussian_filter(
                    values.astype(float), sigma=e_res)

                if "K-40" in name:
                    out[i] = delta(spect, mod=1)
                elif "Th-232" in name:
                    out[i] = delta(spect)
                else:
                    out[i] = delta(spect, mod=2)

            return out

        scale_data = RealData(np.arange(3), np.array([k_pos, tl_pos, pb_pos]))
        scale_model = Model(get_scale)
        scale_odr = ODR(scale_data, scale_model, beta0=[2, 7, 5])
        scale_odr.set_job(fit_type=2)
        scale_output = scale_odr.run()
        print(np.abs(scale_output.beta))

        params = [1 + 2e-3, 7e-7, 0.3]
        if plot_index == 0:
            if bulk:
                params = [1 + 2.5e-4, 6.3e-7, 0.4]
            else:
                params = [1 + 8.2e-4, 8e-7, 0.3]
        elif plot_index == 1:
            params = [1 + 2.5e-3, 8e-7, 0.6]
        elif plot_index == 2:
            params = [1 + 2.5e-3, 7e-7, 0.3]
        elif plot_index == 5:
            params = [1 + 2.5e-3, 7e-7, 0.3]
        elif plot_index == 6:
            params = [1 + 2.7e-3, 7.5e-7, 0.5]

        for name in sim_names:
            values, edges = np.histogram(
                params[0] * _sim_data[name.split("_")[0]] -
                params[1] * _sim_data[name.split("_")[0]]**2 + params[2],
                bins=LENGTH, range=(xdata[0], xdata[-1] + ENERGY_PER_BIN))

            spect = scipy.ndimage.gaussian_filter(values.astype(float), sigma=e_res)
            #print(edges)
            #print(len(edges))

            #sim_data[name] = np.load(os.path.join(
            #    s_path, name + "_Counts.npy"))#[:de][CUT]
            sim_data[name] = 1e3 * spect / _sim_norm[name.split("_")[0]]

            if "K-40" in name:
                k_pos_0 = delta(sim_data[name], mod=1)
            elif "Th-232" in name:
                tl_pos_0 = delta(sim_data[name])
            else:
                pb_pos_0 = delta(sim_data[name], mod=2)

        #de = int(np.round(
        #    (tl_pos_0 - tl_pos + k_pos_0 - k_pos + pb_pos_0 - pb_pos) /
        #    (ENERGY_PER_BIN * 3), 0))
        k_de = int((k_pos_0 - k_pos) / ENERGY_PER_BIN)
        tl_de = int((tl_pos_0 - tl_pos) / ENERGY_PER_BIN )
        pb_de = int((pb_pos_0 - pb_pos) / ENERGY_PER_BIN)
        print(pb_de)
        print(k_de)
        print(tl_de)

        keys = sim_names
        deltas = {keys[0]: k_de, keys[1]: tl_de, keys[2]: pb_de}
        min_e = {keys[0]: 1425, keys[1]: 2550, keys[2]: 340}
        max_e = {keys[0]: 1495, keys[1]: 2680, keys[2]: 354}
        data_x = {}
        data_y = {}
        sim_y = {}
        cuts = {}
        bg_sim = {}
        for (key, de) in deltas.items():
            if de <= 0:
                #data_x[key] = xdata[1-de:]
                #data_y[key] = data[1-de:]
                data_x[key] = xdata#[-de:]
                data_y[key] = data#[-de:]
                cuts[key] = (data_x[key] < max_e[key]) & \
                        (data_x[key] > min_e[key])
                cuts[key] &= (data_y[key] > 0) & (data_x[key] >= 100)
                #sim_y[key] = sim_data[key][:de][cuts[key]]
                sim_y[key] = sim_data[key][cuts[key]]
            else:
                #data_x[key] = xdata[:-de-1]
                #data_y[key] = data[:-de-1]
                data_x[key] = xdata#[:-de]
                data_y[key] = data#[:-de]
                cuts[key] = (data_x[key] < max_e[key]) & \
                        (data_x[key] > min_e[key])
                cuts[key] &= (data_y[key] > 0) & (data_x[key] >= 100)
                #sim_y[key] = sim_data[key][de:][cuts[key]]
                sim_y[key] = sim_data[key][cuts[key]]

            #print(data_x[key])
            if key == keys[2]:
                bg_x = data_x[key].copy()
                bg_y = 1e3 * data_y[key] / norm
                bg_cut = (data_y[key] > 0) & (bg_x >= 100)
                bg_cut &= (bg_x <= 505.) | (bg_x >= 515.)
                bg_cut &= (bg_x <= 1170.) | (bg_x >= 1175.)
                bg_cut &= (bg_x <= 1328.) | (bg_x >= 1335.)

                for name in sim_names:
                    if de < 0:
                        bg_sim[name] = sim_data[name]#[:de]
                    else:
                        bg_sim[name] = sim_data[name]#[de:]
                    #bg_cut &= bg_sim[name] > bg_sim[name][bg_sim[name] > 0].min()

                for name in sim_names:
                    bg_sim[name] = bg_sim[name][bg_cut]

            data_x[key] = data_x[key][cuts[key]]
            data_y[key] = 1e3 * data_y[key][cuts[key]] / norm

        #CUT_K = (sim_x[CUT] < 1485) & (sim_x[CUT] > 1435)
        #CUT_TL = (sim_x[CUT] < 2630) & (sim_x[CUT] > 2600)



        # Moyal (Landau) distribution
        #def moyal(xdata, b0, b1, b2, dy, dx):
        #    x = (xdata - dx) / b1
        #    return b0 * np.exp(-(x + np.exp(-x)) / 2.) + dy

        #def con_fit(xdata,  a0, a1, a2, b0, b1, dy, dx, c0, c1, d0, d1):
        #    x = (xdata - dx) / b1
        #    out = 10. * (b0 * np.exp(-(x + np.exp(-x)) / 2.) - dy)
        #    out += c0 * xdata * np.exp(-np.sqrt(xdata) / c1)
        #    out += d0  * np.log(0.1 * np.abs(xdata - 100) * d1)
        #    _as = [a0, a1, a2]
        #    for i, a in enumerate(_as):
        #        out += a * sim_data[sim_names[i]]
        #    return 1e-2 * out

        mult = 1e-2
        def k_fit(B, x):
            out = 1e-2 * B[1]  - 1e-6 * B[2] * x
            out += mult * B[0] * sim_y[keys[0]]
            return out

        def tl_fit(B, x):
            out = 1e-3 * B[1]  - 1e-7 * B[2] * x
            out += mult * B[0] * sim_y[keys[1]]
            return out

        def pb_fit(B, x):
            out = 1e-1 * B[1]  - 1e-4 * B[2] * x
            out += mult * B[0] * sim_y[keys[2]]
            return out

        SBOUNDS = [0, 1000]

        P0_BG = [70, 51, 158, 13, 18, 33, 20]
        B0_BG = [1, 1, 10, 0, 1, 0, 1]
        B1_BG = [1000, 1000, 1000, 1000, 1000, 1000, 1000]


        #P0 = [0.1] + P0_BG
        #BOUNDS = np.array([[0] + B0_BG, [np.inf] + B1_BG])

        P0 = P0_BG
        BOUNDS = np.array([B0_BG, B1_BG])


    def get_fit(mod=0):
        if (mod == 0):
            title = tex_to_uni("^{208}Tl 583 keV: ")
            my_range = (xdata >= 578) & (xdata <= 588)
            color = 1
        elif (mod == 1):
            title = tex_to_uni("^{208}Tl 2615 keV: ")
            my_range = (xdata >= 2605) & (xdata <= 2625)
            color = 1
        elif (mod == 2):
            title = tex_to_uni("^{60}Co 1173 keV: ")
            my_range = (xdata >= 1163) & (xdata <= 1183)
            color = 2
        elif (mod == 3):
            title = tex_to_uni("^{60}Co 1332 keV: ")
            my_range = (xdata >= 1322) & (xdata <= 1342)
            color = 2
        elif (mod == 4):
            title = tex_to_uni("^{40}K   1461 keV: ")
            my_range = (xdata >= 1451) & (xdata <= 1471)
            color = 3
        elif (mod == 5):
            title = tex_to_uni("^{212}Pb 239 keV: ")
            my_range = (xdata >= 234) & (xdata <= 244)
            color = 4
        elif (mod == 6):
            title = tex_to_uni("^{214}Pb 352 keV: ")
            my_range = (xdata >= 347) & (xdata <= 357)
            color = 5
        elif (mod == 7):
            title = tex_to_uni("^{214}Bi 609 keV: ")
            my_range = (xdata >= 604) & (xdata <= 614)
            color = 6
        if peak_info:
            print(title)
        fit_x = xdata[my_range]
        fit_y = data[my_range]

        gauss_p0 = [fit_y.max() / 2., fit_x[fit_y == fit_y.max()][0], 5e-1, 5e-5, -1e-8]
        bounds = [[0., 0., 0.1, -np.inf, -np.inf], [np.inf, 3e3, 10, np.inf, 0]]
        try:
            popt, cov = curve_fit(gauss_fit, fit_x, fit_y, p0=gauss_p0, bounds=bounds)
            dE_out = int(popt[1])

            if (popt[1] - dE_out >= 0.5):
                dE_out += 1
            if peak_info:
                print(tex_to_uni("\deltaE\,=\," + str(dE_out) + "\,keV"))
                print("amplitude: " + tex_to_uni(str(np.round(1e3 * popt[0] / norm, 3)) +
                        "\,\times\,10^{-3}\,cts\,s^{-1}"))


            out_number = 1e3 * get_area(popt) / (norm * (xdata[1] - xdata[0]))
            out_number_error = 1e3 * get_area_error(popt, cov) / (norm * (xdata[1] - xdata[0]))
            out_number_error += 1e3 * np.sqrt(get_area(popt)) / (norm * (xdata[1] - xdata[0]))

            if (popt[0] / (popt[3] + popt[4] * popt[1])) < 0.1:
                out_number = fit_y.sum() * 1e3 / norm
                out_number_error = 0

            dec = get_dec(out_number_error)

            out_str = "(" + get_string(out_number, dec=dec) + \
                    tex_to_uni("\,\pm\,") + \
                    get_string(out_number_error, dec=dec) + ")"

            if out_number_error == 0:
                out_str = "< " + get_string(out_number)
            out_str += tex_to_uni("\,\times\,10^{-3}\,cts\,s^{-1}")

            if peak_info:
                print(out_str)

            fit = ptool.Curve(
                fit_x, gauss_fit(fit_x, *popt),
                title=title + out_str, color=color)
        except RuntimeError:
            fit = ptool.Curve(np.array([0, 0]), np.array([0.01, 0.01]), alpha=0.)
        return fit

    sum_fit = get_fit() * get_fit(1) * get_fit(2) * get_fit(3)
    sum_fit *= get_fit(4) * get_fit(5) * get_fit(6)  * get_fit(7)

    #(out * sum_fit).plot()

    if sim_path is not None:
        k_data = RealData(data_x[keys[0]], data_y[keys[0]], sy=np.sqrt(data_y[keys[0]]), sx=2.5)
        k_model = Model(k_fit)
        k_odr = ODR(k_data, k_model, beta0=[7, 2, 2])
        k_odr.set_job(fit_type=2)
        k_output = k_odr.run()
        k_popt = k_output.beta
        k_cov = k_output.sd_beta

        tl_data = RealData(data_x[keys[1]], data_y[keys[1]], sy=np.sqrt(data_y[keys[1]]), sx=2.5)
        tl_model = Model(tl_fit)
        tl_odr = ODR(tl_data, tl_model, beta0=[7, 2, 2])
        tl_odr.set_job(fit_type=2)
        tl_output = tl_odr.run()
        tl_popt = tl_output.beta
        tl_cov = tl_output.sd_beta

        pb_data = RealData(data_x[keys[2]], data_y[keys[2]], sy=np.sqrt(data_y[keys[2]]), sx=2.5)
        pb_model = Model(pb_fit)
        pb_odr = ODR(pb_data, pb_model, beta0=[7, 2, 2])
        pb_odr.set_job(fit_type=2)
        pb_output = pb_odr.run()
        pb_popt = pb_output.beta
        pb_cov = pb_output.sd_beta

        #k_popt, k_cov = curve_fit(
        #    k_fit, data_x[keys[0]], data_y[keys[0]], bounds=SBOUNDS)
        #print(k_popt)
        #tl_popt, tl_cov = curve_fit(
        #    tl_fit, data_x[keys[1]], data_y[keys[1]], bounds=SBOUNDS)
        #print(tl_popt)
        #pb_popt, pb_cov = curve_fit(
        #    pb_fit, data_x[keys[2]], data_y[keys[2]], bounds=SBOUNDS)
        #print(pb_popt)

        def con_fit(B, x):
            B = np.abs(B)
            x0 = (x - B[2]) / B[1]
            out = 1e-2 * B[0] * np.exp(-(x0 + np.exp(-x0)) / 2.)
            out += 1e-3 * B[3] * x * np.exp(-1e-2 * np.sqrt(x) * B[4])
            out += 1e-5 * B[5]  * np.log(1e4 * np.abs(x - 100) * B[6])

            #out += mult * bg_sim[sim_names[0]] * 10
            #out += mult * bg_sim[sim_names[1]] * 10
            #out += mult * bg_sim[sim_names[2]] * 10
            out += mult * k_popt[0] * bg_sim[sim_names[0]]
            out += mult * tl_popt[0] * bg_sim[sim_names[1]]
            out += mult * pb_popt[0] * bg_sim[sim_names[2]]
            return out

        my_data = RealData(bg_x[bg_cut], bg_y[bg_cut], sy=np.sqrt(bg_y[bg_cut]), sx=2.5)
        model = Model(con_fit)

        #P0 = P0_BG + [k_popt[0], tl_popt[0], pb_popt[0]]
        P0 = P0_BG
        odr = ODR(my_data, model, beta0=P0)
        odr.set_job(fit_type=2)
        output = odr.run()
        yn = norm * con_fit(output.beta, bg_x[bg_cut]) / 1e3
        #print(P0[-2:])
        #print([k_popt[0], tl_popt[0], pb_popt[0]])
        #print(np.abs(output.beta[-2:]))

        #popt = output.beta[-3:]
        #cov = output.sd_beta[-3:]
        popt = [k_popt[0], tl_popt[0], pb_popt[0]]
        cov = np.sqrt([k_cov[0], tl_cov[0], pb_cov[0]])
        confit = ptool.Curve(bg_x[bg_cut], yn, title="Fit", color=1, log=True)
        #confit = ptool.Curve(bg_x[bg_cut], mult * norm * bg_sim[sim_names[2]] * pb_popt[0] / 1e3,
        #                    title="Fit", color=1, log=True)
        (out * confit).plot()


        if bulk:
            print("Bulk contamination")
            print("-----------------------")
            for i, num in enumerate(popt):
                #if np.sqrt(cov[i][i]) > num:
                #    continue
                _error = np.round(mult * cov[i] / mass, 2)
                _val = np.round(mult * num / mass, 2)

                print(sim_names[i] + ": " + str(_val) +
                    tex_to_uni("\,\pm\,") + str(_error) +
                    tex_to_uni("\,Bq\,kg^{-1}") )
                print("Limit < " + str(np.round(2 * _error, 4)) +
                    tex_to_uni("\,Bq\,kg^{-1}"))
        else:
            print("Surface contamination")
            print("-----------------------")
            mult *= 1e3
            for i, num in enumerate(popt):
                #if np.sqrt(cov[i][i]) > num:
                #    continue
                _error = np.round(mult * cov[i] / area, 2)
                _val = np.round(mult * num / area, 2)
                print(sim_names[i] + ": " + str(_val) +
                    " +/- " + str(_error) + tex_to_uni("\,mBq\,cm^{-2}"))
                print("Limit < " + str(np.round(2 * _error, 3)) +
                     tex_to_uni("\,mBq\,cm^{-2}"))
