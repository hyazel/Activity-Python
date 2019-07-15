#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:48:23 2019

@author: laurent.droguet
"""

import turicreate as tc
import glob2
import os
import fnmatch

data_dir = '/Users/laurent.droguet/Desktop/Dev/_ML/Activity/Data/**/*.csv'

acc_files = glob2.glob(data_dir, recursive = True)

configfiles = [os.path.join(data_dir, f)
    for dirpath, dirnames, files in os.walk(data_dir)
    for f in fnmatch.filter(files, '*.csv')]

# Load data
target_map = {
    0.: 'Chilling',       
    1.: 'walking',          
    2.: 'Metro'
}

data = tc.SFrame()

for i, xpName in enumerate(acc_files):
    sfRaw = tc.SFrame.read_csv(xpName, delimiter=',', header = True, verbose=False)
    sf = sfRaw['label','motionRotationRateX', 'motionRotationRateY', 'motionRotationRateZ', 
            'motionUserAccelerationX', 'motionUserAccelerationY', 'motionUserAccelerationZ']
    sf['exp_id'] = [i] * sf.num_rows()
    data = data.append(sf)
    
data['activity'] = data['label'].apply(lambda x: target_map[x])
data = data.remove_column('label')

#data.save('activities.sframe') 

train, test = tc.activity_classifier.util.random_split_by_session(data, session_id='exp_id', fraction=0.8)

# Create an activity classifier
model = tc.activity_classifier.create(train, session_id='exp_id', target='activity', prediction_window= 30)

# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test)
print(metrics['accuracy'])

# Save the model for later use in Turi Create
model.save('activities.model')

# Export for use in Core ML
model.export_coreml('Model/MyActivityClassifier.mlmodel')


#sf['activity'] = sf['label'].apply(lambda x: target_map[x])
#data = data.remove_column('label')