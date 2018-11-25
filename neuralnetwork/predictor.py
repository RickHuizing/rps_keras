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


# p, q, r, s
def get_prediction(img):
    pred = model.predict(img).tolist()
    pred_dict_list = []
    for file in pred:
        pred_dict_list.append({'r': file[2], 'p': file[0], 's': file[3], 'q': file[1]})
    #print(pred_dict_list)

    highest_values = [0]*len(pred_dict_list)
    predicted_classes = ['q']*len(pred_dict_list)
    current_img = 0
    for img in pred_dict_list:
        for p_class, p_value in img.items():
            if p_value > highest_values[current_img]:
                highest_values[current_img] = p_value
                predicted_classes[current_img] = p_class
        current_img += 1
    return predicted_classes
