import os
from keras.models import load_model

model_location = os.getcwd() + '\\neuralnetwork\\models\\'
file_list = os.listdir(model_location)
select_string = ''
print('select model to load')
for index, file in enumerate(file_list):
    print(str(index) + ': ' + file)
selected_file = input(select_string)
model = load_model(model_location + file_list[int(selected_file)])


def get_prediction(img):
    pred = model.predict(img).tolist()
    # pred = {'r': pred[0], 'p': pred[0], 's': pred[0], 'nothing': pred[0]}
    pred_dict_list = []
    for file in pred:
        pred_dict_list.append({'r': file[0], 'p': file[1], 's': file[2]})
        # pred_dict_list.append({'r': file[0], 'p': file[1], 's': file[2], , 'nothing': pred[0]})
    return pred_dict_list
