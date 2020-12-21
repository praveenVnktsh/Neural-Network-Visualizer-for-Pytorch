import cv2
import numpy as np
import matplotlib.cm as cm
import torch.nn as nn
from matplotlib.colors import Normalize


def getColor(weight):
    cmap = cm.tab20c
    norm = Normalize(vmin=-1, vmax=1)
    ret = np.round(np.array(cmap(norm(weight)))*255)[0:3]
    return ret


class Visualizer:
    
    @staticmethod
    def draw(nnstatedict, scale = 3):
        statedict=  nnstatedict
        architecture = []
        weights = []
        biases = []
        for key in statedict.keys():    
            if "bias" not in key:
                architecture.append(statedict[key].size()[1])
                weights.append(statedict[key].cpu().numpy().tolist())
            else:
                biases.append(statedict[key].cpu().numpy().tolist())
        
        nnDepth = len(architecture)
        maxNeuronsPerLayer = max(architecture)

        width = nnDepth * 50 * scale
        height =  maxNeuronsPerLayer * 20 * scale

        image = np.zeros((height,width, 3), dtype=np.uint8)
        
        boxHeight =  int(height / (architecture[np.argmax(architecture)]))
        radius = int(boxHeight / 4)

        centers = [[] for i in range(len(architecture))]

        for j in range(len(architecture)):
            initPadding = int(((max(architecture) - architecture[j])/2)*boxHeight)
            for i in range(architecture[j]):
                x = int(width*j/len(architecture)) + int(width/(2*len(architecture)))
                y = initPadding + boxHeight * i + 2 * radius
                centers[j].append((x, y))
                
        
        for i in range(len(centers) - 1):
            for j in range(len(centers[i])):
                for k in range(len(centers[i+1])):
                    weight = weights[i][k][j]
                    # print(weight)
                    color = getColor(weight)
                    cv2.line(image, centers[i][j], centers[i+1][k],color , 1, lineType=cv2.LINE_AA)

        for j in range( len(centers)):
            for i in range(len(centers[j])):
                
                if j == 0:
                    color = (0, 128, 255)
                else:
                    bias = biases[j-1][i]
                    color = getColor(bias)
                cv2.circle(image, centers[j][i], radius, color, -1, lineType=cv2.LINE_AA)
        
        return image


if __name__ == "__main__":
    
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.neuralnetwork = nn.Sequential( 
                nn.Linear(6,12, bias = True),
                nn.ReLU(),  
                nn.Linear(12, 3, bias = True),
                nn.ReLU(),
                nn.Linear(3, 8, bias = True),
                nn.ReLU(),
                nn.Linear(8, 3, bias = True),
                nn.ReLU(),
                nn.Linear(3, 3, bias = True),
                nn.ReLU(),
            ) 

    

    while cv2.waitKey(10) != ord('q'):
        neuralnet = Net()
        visualizer = Visualizer.draw(neuralnet.state_dict(), scale = 2)
        cv2.imshow('NN', visualizer)