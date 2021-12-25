callback_names = {'MD':'Mahalanobis Distance',
'NN Quadratic':'Nearest 10 Neighbours Class Quadratic 1D Typicality',
'NN Linear':'Nearest 10 Neighbours Class 1D Typicality',
'MSP':'Maximum Softmax Probability',
'NN All Dim Class':'Nearest 10 Neighbours Class Typicality',
'Gram':'Gram',
'ODIN':'ODIN',
'KDE':'KDE',
'CAM':'CAM'} # Ablation

# format of Mahalanobis distance: Repeat MD
repeat_names = {value: f'Repeat {key}' for key,value in callback_names.items()}

desired_key_dict = {callback_names['MD']:['Mahalanobis AUROC OOD','Mahalanobis AUPR OOD','Mahalanobis FPR OOD'],
callback_names['NN Quadratic']:['Normalized One Dim Class Quadratic Typicality KNN - 10 OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 FPR OOD -'],
callback_names['NN Linear']: ['Normalized One Dim Class Typicality KNN - 10 OOD -','Normalized One Dim Class Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Typicality KNN - 10 FPR OOD -'],
callback_names['MSP']: ['Maximum Softmax Probability AUROC OOD','Maximum Softmax Probability AUPR OOD','Maximum Softmax Probability FPR OOD'],
callback_names['NN All Dim Class']:['Normalized All Dim Class Typicality KNN - 10 OOD -','Normalized All Dim Class Typicality KNN - 10 AUPR OOD -','Normalized All Dim Class Typicality KNN - 10 FPR OOD -'],
callback_names['Gram']:['Gram AUROC OOD','Gram AUPR OOD','Gram FPR OOD'],
callback_names['ODIN']:['ODIN AUROC OOD','ODIN AUPR OOD','ODIN FPR OOD'],
callback_names['KDE']:['KDE AUROC OOD','KDE AUPR OOD','KDE FPR OOD'],
callback_names['CAM']:[],
} # Callback information for ablation


# could potentially make a separate callback dict for the case of the ablations