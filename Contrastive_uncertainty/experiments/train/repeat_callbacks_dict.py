from pytorch_lightning import callbacks


callback_names = {'MD':'Mahalanobis Distance',
'NN Quadratic':'Nearest 10 Neighbours Class Quadratic 1D Typicality',
'NN Linear':'Nearest 10 Neighbours Class 1D Typicality',
'MSP':'Maximum Softmax Probability',
'NN All Dim Class':'Nearest 10 Neighbours Class Typicality',
'Gram':'Gram',
'ODIN':'ODIN',
'KDE':'KDE',
'CAM':'CAM',
'Total Centroid KL':'Total Centroid KL', # Ablation
'Feature Entropy':'Feature Entropy',
# Marginal Quadratic Typicality
'NN Marginal Quadratic':'Nearest 10 Neighbours Marginal Quadratic 1D Typicality',
# Quadratic typicality with a single sample
'NN Quadratic Single':'Nearest 1 Neighbours Class Quadratic 1D Typicality',
'NN Quadratic 1D Scores':'Nearest 10 Neighbours Class 1D Scores Typicality',
'Analysis NN Quadratic 1D Scores':'Nearest 10 Neighbours Analysis Class 1D Scores Typicality',
'KL Distance':'KL Distance OOD',
'Metrics':'Metrics'}


# format of Mahalanobis distance: Repeat MD, used to choose whether to repeat a callback or not
repeat_names = {value: f'Repeat {key}' for key,value in callback_names.items()}

desired_key_dict = {callback_names['MD']:['Mahalanobis AUROC OOD','Mahalanobis AUPR OOD','Mahalanobis FPR OOD'],
callback_names['NN Quadratic']:['Normalized One Dim Class Quadratic Typicality KNN - 10 OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 FPR OOD -'],
callback_names['NN Linear']: ['Normalized One Dim Class Typicality KNN - 10 OOD -','Normalized One Dim Class Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Typicality KNN - 10 FPR OOD -'],
callback_names['MSP']: ['Maximum Softmax Probability AUROC OOD','Maximum Softmax Probability AUPR OOD','Maximum Softmax Probability FPR OOD'],
callback_names['NN All Dim Class']:['Normalized All Dim Class Typicality KNN - 10 OOD -','Normalized All Dim Class Typicality KNN - 10 AUPR OOD -','Normalized All Dim Class Typicality KNN - 10 FPR OOD -'],
callback_names['Gram']:['Gram AUROC OOD','Gram AUPR OOD','Gram FPR OOD'],
callback_names['ODIN']:['ODIN AUROC OOD','ODIN AUPR OOD','ODIN FPR OOD'],
callback_names['KDE']:['KDE AUROC OOD','KDE AUPR OOD','KDE FPR OOD'],
callback_names['NN Marginal Quadratic']:['Normalized One Dim Marginal Quadratic Typicality KNN - 10 OOD -','Normalized One Dim Marginal Quadratic Typicality KNN - 10 AUPR OOD -','Normalized One Dim Marginal Quadratic Typicality KNN - 10 FPR OOD -'], 
callback_names['NN Quadratic Single']:['Normalized One Dim Class Quadratic Typicality KNN - 1 OOD -','Normalized One Dim Class Quadratic Typicality KNN - 1 AUPR OOD -','Normalized One Dim Class Quadratic Typicality KNN - 1 FPR OOD -'],
callback_names['NN Quadratic 1D Scores']:['Normalized One Dim Scores Class Quadratic Typicality KNN - 10 OOD -'],
callback_names['Analysis NN Quadratic 1D Scores']:['Analysis Normalized One Dim Scores Class Quadratic Typicality KNN - 10 OOD -'],
callback_names['KL Distance']:['KL Distance OOD'],
} # Callback information for ablation

# Key dict only for the ID data
desired_ID_key_dict = {callback_names['CAM']:['GradCam Heatmaps'],
callback_names['Total Centroid KL']:['KL Divergence(Total||Class)'],
callback_names['Feature Entropy']:['Marginal Feature Entropy','Class Conditional Feature Entropy'],
callback_names['Metrics']:['rho_spectrum@']}

# could potentially make a separate callback dict for the case of the ablations