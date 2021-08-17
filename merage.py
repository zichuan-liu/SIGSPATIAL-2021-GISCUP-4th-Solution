import pandas as pd
import numpy as np

sub = pd.read_csv('./subs/12205.csv')

cv1 = pd.read_csv('./subs/12205.csv')
cv2 = pd.read_csv('./subs/12229_sig.csv')
cv4 = pd.read_csv('./subs/sub_0.132646.csv')
cv5 = pd.read_csv('./subs/pred_2021_07_24_09_40.csv')

sub['result'] = cv1['result']*0.43 + cv2['result']*0.43 + cv5['result']*0.1+ cv4['result']*0.04
sub.to_csv('subs/tf-43__tf2-43__troch-10__lgb-4.csv',index=False)
