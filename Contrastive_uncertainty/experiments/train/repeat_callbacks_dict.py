callback_names = {'MD':'Mahalanobis Distance',
'NN Quadratic':'Nearest 10 Neighbours Class Quadratic 1D Typicality',
'NN Linear':'Nearest 10 Neighbours Class 1D Typicality',
'MSP':'Maximum Softmax Probability'}


# format of Mahalanobis distance: Repeat MD
repeat_names = {value: f'Repeat {key}' for key,value in callback_names.items()}

desired_key_dict = {callback_names['MD']:['Mahalanobis AUROC OOD','Mahalanobis AUPR OOD','Mahalanobis FPR OOD'],
callback_names['NN Quadratic']:['Normalized One Dim Class Quadratic Typicality KNN - 10 OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Quadratic Typicality KNN - 10 FPR OOD -'],
callback_names['NN Linear']: ['Normalized One Dim Class Typicality KNN - 10 OOD -','Normalized One Dim Class Typicality KNN - 10 AUPR OOD -','Normalized One Dim Class Typicality KNN - 10 FPR OOD -'],
callback_names['MSP']: ['Maximum Softmax Probability AUROC OOD','Maximum Softmax Probability AUPR OOD','Maximum Softmax Probability FPR OOD']}
