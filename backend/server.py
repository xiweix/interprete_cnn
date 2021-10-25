import os
import gzip
import asyncio
import websockets
import base64
import time
from threading import Thread
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from mnist_test import Model, mnist_main, get_activations
from utils import make_outputdir, get_dataset_from_np


train_set, test_set = get_dataset_from_np()


async def hello(websocket, path):
    await websocket.send('READY')
    stop = False
    while stop == False:
        rec_m = await websocket.recv()
        if 'start' in rec_m:
            _, n_layers1, features1, drop1, lr1, n_layers2, features2, drop2, lr2, epochs, train_batch_size, lr_step_gamma = rec_m.split(
                '***')
            n_layers1 = int(n_layers1)
            drop1 = float(drop1)
            lr1 = float(lr1)
            n_layers2 = int(n_layers2)
            drop2 = float(drop2)
            lr2 = float(lr2)
            epochs = int(epochs)
            train_batch_size = int(train_batch_size)
            lr_step_gamma = float(lr_step_gamma)
            if n_layers1 > 1:
                features1 = features1.split(',')
                features1 = list(map(int, features1))
            else:
                features1 = [int(features1)]
            if n_layers2 > 1:
                features2 = features2.split(',')
                features2 = list(map(int, features2))
            else:
                features2 = [int(features2)]
            outputdir, timestamp = make_outputdir()
            await websocket.send('start_training***')
            print(
                f'Message received. Start training now. Results saved in {timestamp}')
            t1 = Thread(target=mnist_main, args=(epochs, train_batch_size, lr_step_gamma, n_layers1,
                                                 features1, drop1, lr1, train_set, test_set, outputdir, 'model1'), daemon=True)
            t2 = Thread(target=mnist_main, args=(epochs, train_batch_size, lr_step_gamma, n_layers2,
                                                 features2, drop2, lr2, train_set, test_set, outputdir, 'model2'), daemon=True)
            t1.start()
            t2.start()
        elif 'refresh' in rec_m:
            if os.path.exists(os.path.join(outputdir, 'model1', 'train_loss.npy')) and os.path.exists(os.path.join(outputdir, 'model2', 'train_loss.npy')):
                model1_train_loss = np.load(os.path.join(
                    outputdir, 'model1', 'train_loss.npy')).tolist()
                model1_train_acc = np.load(os.path.join(
                    outputdir, 'model1', 'train_acc.npy')).tolist()
                model1_test_loss = np.load(os.path.join(
                    outputdir, 'model1', 'test_loss.npy')).tolist()
                model1_test_acc = np.load(os.path.join(
                    outputdir, 'model1', 'test_acc.npy')).tolist()
                model2_train_loss = np.load(os.path.join(
                    outputdir, 'model2', 'train_loss.npy')).tolist()
                model2_train_acc = np.load(os.path.join(
                    outputdir, 'model2', 'train_acc.npy')).tolist()
                model2_test_loss = np.load(os.path.join(
                    outputdir, 'model2', 'test_loss.npy')).tolist()
                model2_test_acc = np.load(os.path.join(
                    outputdir, 'model2', 'test_acc.npy')).tolist()
                length = min([len(model1_test_acc), len(model2_test_acc)])
                epoch_list = list(range(1, length + 1))
                sendMsg = f"plotLossTrain***{epoch_list}***{model1_train_loss[0:length]}***{model2_train_loss[0:length]}"
                await websocket.send(sendMsg)
                sendMsg = f"plotAccTrain***{epoch_list}***{model1_train_acc[0:length]}***{model2_train_acc[0:length]}"
                await websocket.send(sendMsg)
                sendMsg = f"plotLossTest***{epoch_list}***{model1_test_loss[0:length]}***{model2_test_loss[0:length]}"
                await websocket.send(sendMsg)
                sendMsg = f"plotAccTest***{epoch_list}***{model1_test_acc[0:length]}***{model2_test_acc[0:length]}"
                await websocket.send(sendMsg)
            if os.path.exists(os.path.join(outputdir, 'model1', f'epoch.{epochs}.test_acc.npy')) and os.path.exists(os.path.join(outputdir, 'model2', f'epoch.{epochs}.test_acc.npy')):
                await websocket.send('train_finished***')
        elif 'request_img' in rec_m:
            Index = int(rec_m.split('***')[1])
            print('request image: ', Index)
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            test_set_np = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', 'mnist_test_data_b.npy'))
            test_set_target_np = np.load(os.path.join(os.path.dirname(os.getcwd()), 'data', 'mnist_test_target.npy'))
            sample_data = test_set_np[Index]
            np.save(os.path.join(temp_dir, 'org.npy'), sample_data)
            sample_label = test_set_target_np[Index]
            sample_img = Image.fromarray(sample_data)
            sample_img.save(os.path.join(temp_dir, 'org.png'))
            with open(os.path.join(temp_dir, 'org.png'), 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            sendMsg = "sample_img***" + img_data + '***' + str(sample_label)
            await websocket.send(sendMsg)
            sendMsg = "sample_canvas***" + img_data + '***' + str(sample_label)
            await websocket.send(sendMsg)
            await websocket.send("clearActivations1***")
            await websocket.send("clearActivations2***")
            print('Send the requested original image to the front end')
        elif 'request_activations' in rec_m:
            selected_epoch = int(rec_m.split('***')[1])
            await websocket.send('epochSelected***')
            print(
                f'received request to load mode from epoch {selected_epoch}. Start to generate activations')
            model1_path = os.path.join(
                outputdir, 'model1', f'epoch.{selected_epoch}.pt.gz')
            model2_path = os.path.join(
                outputdir, 'model2', f'epoch.{selected_epoch}.pt.gz')
            temp_dir = os.path.join(os.getcwd(), 'temp')
            sample_data = np.load(os.path.join(temp_dir, 'org.npy'))
            model1_actname, prediction1 = get_activations(
                model1_path, n_layers1, features1, drop1, sample_data, 'model1', 'org')
            model2_actname, prediction2 = get_activations(
                model2_path, n_layers2, features2, drop2, sample_data, 'model2', 'org')
            sendMsg = f"model1Activations***{len(model1_actname)}***{prediction1}***{selected_epoch}"
            for i in range(len(model1_actname)):
                act_name = model1_actname[i]
                with open(os.path.join(temp_dir, f'model1.org_{act_name}.png'), 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                sendMsg += f'***{act_name}***{img_data}'
            await websocket.send(sendMsg)
            sendMsg = f"model2Activations***{len(model2_actname)}***{prediction2}***{selected_epoch}"
            for i in range(len(model2_actname)):
                act_name = model2_actname[i]
                with open(os.path.join(temp_dir, f'model2.org_{act_name}.png'), 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                sendMsg += f'***{act_name}***{img_data}'
            await websocket.send(sendMsg)
            print(
                f'Model1 Org activation sent. Prediction: {prediction1}. Epoch: {selected_epoch}')
            print(
                f'Model2 Org activation sent. Prediction: {prediction2}. Epoch: {selected_epoch}')
        elif 'request_annotated_activations' in rec_m:
            annotated_img = rec_m.split('***')[1]
            prefix = "data:image/png;base64,"
            annotated_img = annotated_img[len(prefix):]
            if len(annotated_img) % 4 != 0:
                annotated_img += '=' * (4 - len(annotated_img) % 4)
            annotated_data = base64.b64decode(annotated_img)
            temp_dir = os.path.join(os.getcwd(), 'temp')
            annotated_path = os.path.join(temp_dir, 'annotated.png')
            if os.path.exists(annotated_path):
                os.remove(annotated_path)
            with open(annotated_path, "wb") as f:
                f.write(annotated_data)
            annotated_input_img = Image.open(
                annotated_path).convert('L').resize((28, 28))
            annotated_input_data = np.asarray(annotated_input_img)
            # print('annotated_input_img: ', annotated_input_data.shape)
            annotated_input_data = np.copy(annotated_input_data)
            np.save(os.path.join(temp_dir, 'annotated.npy'),
                    annotated_input_data)
            model1_actname, prediction1 = get_activations(
                model1_path, n_layers1, features1, drop1, annotated_input_data, 'model1', 'annotated')
            model2_actname, prediction2 = get_activations(
                model2_path, n_layers2, features2, drop2, annotated_input_data, 'model2', 'annotated')
            sendMsg = f"model1AnnotationActivations***{len(model1_actname)}***{prediction1}***{selected_epoch}"
            for i in range(len(model1_actname)):
                act_name = model1_actname[i]
                with open(os.path.join(temp_dir, f'model1.annotated_{act_name}.png'), 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                sendMsg += f'***{act_name}***{img_data}'
            await websocket.send(sendMsg)
            sendMsg = f"model2AnnotationActivations***{len(model2_actname)}***{prediction2}***{selected_epoch}"
            for i in range(len(model2_actname)):
                act_name = model2_actname[i]
                with open(os.path.join(temp_dir, f'model2.annotated_{act_name}.png'), 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                sendMsg += f'***{act_name}***{img_data}'
            await websocket.send(sendMsg)
            print(
                f'Model1 annotated activation sent. Prediction: {prediction1}. Epoch: {selected_epoch}')
            print(
                f'Model2 annotated activation sent. Prediction: {prediction2}. Epoch: {selected_epoch}')

start_server = websockets.serve(hello, "localhost", 6060)


asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
