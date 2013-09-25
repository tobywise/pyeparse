# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pyximport
pyximport.install()
import parsextended
sample = '5565970.0     779.4   448.2   390.0    -0.8     5.9   45.60   ''46.10 32768.0 ...'
print parsextended.parse_samples(sample)
saccade = 'ESACC R  5548035.0  5548320.0   286   843.6   491.3   850.9  423.4    1.48     312   45.60   46.10'
print parsextended.parse_saccades(saccade)
blink = 'EBLINK R 5495553    5495755 203'
print parsextended.parse_blink(blink)
fixation = 'EFIX R   5548321.0  5552261.0   3941      836.7   484.1     622   45.60   46.10'
print parsextended.parse_fixation(fixation)
message = 'MSG 5399829.0 trial-1_start_inc-soc-m_block-15_correct-l-time-2546256'
print parsextended.parse_message(message)
resp = '5551823.0 trial-11_ responses:1 button:1 type:response_other code:1 time:2697320 unc_dms:1'
print parsextended.parse_button_press(resp)
modality = 'MSG 5565983.0 Modality-hand-END'
print parsextended.parse_modality(modality)

# XXX add test file
fname, n_trials = 'exp2/data/OSSRC_108411.asc', 192

parsextended.data_frame_from_log(fname, n_trials)