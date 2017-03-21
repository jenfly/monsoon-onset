#!/bin/bash -v

cp map_01.pdf ..
pdfunite corr_tseries_* ../corr_tseries.pdf
pdfunite index_tseries_* ../index_tseries_all.pdf
pdfunite index_tseries_HOWI_100_* ../index_tseries_HOWI_100.pdf
pdfunite index_tseries_OCI_* ../index_tseries_OCI.pdf
pdfunite index_tseries_SJKE_* ../index_tseries_SJKE.pdf
pdfunite index_tseries_TT_* ../index_tseries_TT.pdf
pdfunite index_tseries_WLH_MERRA_MFC_nroll7_* ../index_tseries_WLH_MERRA_MFC_nroll7.pdf
pdfunite onset_0* onset_yrs_* ../onset.pdf
pdfunite onset_retreat_hist_* ../onset_retreat_hist.pdf
pdfunite strength_* ../strength.pdf
