from SWARNN import SWARNN
from time import sleep
import pandas as pd
import numpy as np


# for month of November 2023
class ExptLog:

    def __init__(self):
        self.fn = None

    def date(self, date):
        default = "Incorrect date"
        self.fn = 'C:\\Users\\DELL\\Documents\\spin_wave_expts\\nov_23\\nov_06_23'
        # date: DD format
        return getattr(self, 'case_' + str(date), lambda: default)()

    def case_06c(self):
        # Near MSSW dipole gap
        fc = 2068e06
        span = 5e06
        rbw = 2e03

        obj = SWARNN(RFGfreq=[2e09], RFGpwr=[10],
                     SAcenterfreq=fc, SAspan=span, SArbw=rbw, SAtracepts=4001, SAref=0)

        X = pd.read_csv(self.fn + '\\dft_irisclassif_allfourfeats_featbin_new.csv', sep=',')

        # All feature frequencies at the same time
        sfn = 'elmswarr_irisclassif_loss15dB_Im1.8A_irisclassif_allfourfeats_featbin_new'

        obj.elm_mixeddrive(savepath=self.fn + '\\' + sfn,
                           f_IF=X.loc[:, 'f_IF0 (MHz)':'f_IF3 (MHz)'].to_numpy() * 1e06,
                           Vpp_IF=X.loc[:, 'Vpp_IF (V)'].to_numpy(),
                           weight=None)

    def case_06b(self):
        # Near MSSW dipole gap
        fc = 2068e06
        span = 5e06
        rbw = 2e03

        obj = SWARNN(RFGfreq=[2e09], RFGpwr=[10],
                     SAcenterfreq=fc, SAspan=span, SArbw=rbw, SAtracepts=4001, SAref=0)

        X = pd.read_csv(self.fn + '\\dft_irisclassif_allfourfeats_featbin_new.csv', sep=',')

        # Feature frequency independently for first instance
        sfn = 'elmswaro_irisclassif_loss4dB_Im1.8A_irisclassif_featseparately_featbin_new'

        freqs = X.loc[0, 'f_IF0 (MHz)':'f_IF3 (MHz)'].to_numpy().reshape(-1, 1)

        obj.elm_mixeddrive(savepath=self.fn + '\\' + sfn,
                           f_IF=freqs * 1e06,
                           Vpp_IF=X.loc[0, 'Vpp_IF (V)'] * np.ones(freqs.shape[0]),
                           weight=None)

        # All feature frequencies at the same time
        sfn = 'elmswaro_irisclassif_loss4dB_Im1.8A_irisclassif_allfourfeats_featbin_new'

        obj.elm_mixeddrive(savepath=self.fn + '\\' + sfn,
                           f_IF=X.loc[:, 'f_IF0 (MHz)':'f_IF3 (MHz)'].to_numpy() * 1e06,
                           Vpp_IF=X.loc[:, 'Vpp_IF (V)'].to_numpy(),
                           weight=None)

    def case_06a(self):
        # Near MSSW dipole gap
        fc = 2068e06
        span = 5e06
        rbw = 2e03
        obj = SWARNN(RFGfreq=[2e09], RFGpwr=[10],
                     SAcenterfreq=fc, SAspan=span, SArbw=rbw, SAtracepts=4001, SAref=0)

        sfn = 'get_mixedsig_irisclassif_allfourfeats_featbin_new'
        X = pd.read_csv(self.fn + '\\dft_irisclassif_allfourfeats_featbin_new.csv', sep=',')
        obj.elm_mixeddrive(savepath=self.fn + '\\' + sfn,
                           f_IF=X.loc[:, 'f_IF0 (MHz)':'f_IF3 (MHz)'].to_numpy() * 1e06,
                           Vpp_IF=X.loc[:, 'Vpp_IF (V)'].to_numpy(),
                           weight=None)


expt = ExptLog()
# date: DD format
expt.date('06c')
