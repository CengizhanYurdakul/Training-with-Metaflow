import os
import torch
from torch.utils.data import DataLoader

from src.Models.ModelFactory import ModelFactory
from src.Datasets.GenderDataset import GenderDataset
from src.Datasets.ValidationDataset import ValidationDataset
from src.Losses.LossFactory import LossFunctionFactory
from src.Optimizers.OptimizerFactory import OptimizerFactory

class Trainer:
    def __init__(self, args=None):
        
        self.args = args
        
        self.initGlobalVariables()
        self.initModel()
        self.initDataset()
        self.initOptimizer()
        self.initLossFunction()
        self.initDevice()
        self.initCheck()
        
    def initGlobalVariables(self):
        self.classifier = None
        self.optimizer = None
        self.lossFunction = None
        
        self.dataLoaderTrain = None
        self.dataLoaderTest = None
        self.dataLoaderValidation = None
        
        self.runNeptune = None
        
        self.falsePredictedIndex = None
        
        self.trainIter = 0
        self.testIter = 0
        self.validationIter = 0
        
        self.testAccuracy = 0
        self.trainAccuracy = 0
        self.validationAccuracy = 0
        
        self.truePredicted = 0
        
        self.bestValidationAccuracy = 0
        
    def initModel(self):
        classifierFactory = ModelFactory()
        self.classifier = classifierFactory.getModel(self.args["backbone"], self.args["pretrained"], self.args["numberClass"])
    
    def initDataset(self):
        datasetTrain = GenderDataset(isTrain=True, seed=self.args["seed"], imagePath=self.args["imagePath"], informationCSV=self.args["informationCSV"], imageSize=self.args["imageSize"], testSize=self.args["testSize"])
        datasetTest = GenderDataset(isTrain=False, seed=self.args["seed"], imagePath=self.args["imagePath"], informationCSV=self.args["informationCSV"], imageSize=self.args["imageSize"], testSize=self.args["testSize"])
        datasetValidation = ValidationDataset(seed=self.args["seed"], validationImagePath=self.args["validationImagePath"], validationInformationCSV=self.args["validationInformationCSV"], imageSize=self.args["imageSize"])
        
        self.dataLoaderTrain = DataLoader(
            datasetTrain,
            batch_size=self.args["batchSize"],
            num_workers=self.args["numWorkers"]
        )
        
        self.dataLoaderTest = DataLoader(
            datasetTest,
            batch_size=self.args["batchSize"],
            num_workers=self.args["numWorkers"]
        )
        
        self.dataLoaderValidation = DataLoader(
            datasetValidation,
            batch_size=self.args["batchSize"],
            num_workers=self.args["numWorkers"]
        )
        
    def initOptimizer(self):
        optimizerFactory = OptimizerFactory()
        self.optimizer = optimizerFactory.getOptimizer(optimizerName=self.args["optimizer"], lr=self.args["lr"], momentum=self.args["momentum"], modelParameters=self.classifier.parameters())
        
    def initLossFunction(self):
        lossFunctionFactory = LossFunctionFactory()
        self.lossFunction = lossFunctionFactory.getLossFunction(self.args["lossFunction"])
    
    def initDevice(self):
        if (torch.cuda.is_available() is False) and ("cuda" in self.args["devices"]):
            raise Exception("CUDA is not available for PyTorch!")
        else:
            self.classifier.to(self.args["devices"])
            self.lossFunction.to(self.args["devices"])
    
    def initCheck(self):
        if (self.classifier is None) or (self.dataLoaderTrain is None) or (self.dataLoaderTest is None) or (self.optimizer is None) or (self.lossFunction is None):
            raise Exception("There are some `None` values in Trainer!")   

    def compareLabels(self, predictedLabels, gtLabels):
        predictedClass = torch.argmax(predictedLabels, 1)
        gtClass = torch.argmax(gtLabels, 1)
        
        self.truePredicted += (gtClass == predictedClass).sum()
        
        self.falsePredictedIndex = gtClass != predictedClass
                        
    def calculateAccuracy(self, type):
        if type == "Train":
            accuracy = (self.truePredicted/(self.dataLoaderTrain.__len__() * self.args["batchSize"])).item()
        elif type == "Test":
            accuracy = (self.truePredicted/(self.dataLoaderTest.__len__() * self.args["batchSize"])).item()
        elif type == "Validation":
            accuracy = (self.truePredicted/(self.dataLoaderValidation.__len__() * self.args["batchSize"])).item()
        return accuracy
    
    def saveModel(self):
        if not os.path.exists(self.args["savePath"]):
            os.makedirs(self.args["savePath"])
            
        torch.save(self.classifier, os.path.join(self.args["savePath"], "PyTorchModel_%s_%s.pth" % (self.args["backbone"], self.epoch)))
        
    def checkSave(self):
        if (self.validationAccuracy > self.bestValidationAccuracy) and (self.validationAccuracy >= self.args["accuracyThreshold"]):
            self.saveModel()
            print("[Current Validation Accuracy: %s > Best Validation Accuracy: %s] Model saved!" % (round(self.validationAccuracy, 3), round(self.bestValidationAccuracy, 3)))
            self.bestValidationAccuracy = self.validationAccuracy
        
    def validate(self):
        self.classifier.eval()
        self.truePredicted = 0
        for c, batch in enumerate(self.dataLoaderValidation):
            inputImage, gtLabel = batch["inputImage"].to(self.args["devices"]), batch["label"].to(self.args["devices"])
            
            predictedLabel = self.classifier(inputImage)
            
            loss = self.lossFunction(predictedLabel, gtLabel)
            
            self.compareLabels(predictedLabel, gtLabel)
            
            if (c % (int(self.dataLoaderValidation.__len__() * self.args["logInterval"])) == 0) and (c != 0):                
                print(
                    "[Validation] - [Epoch: %s / %s] - [Iter: %s / %s] - [Loss: %s]"
                    % (self.epoch, self.args["numEpochs"], c, self.dataLoaderValidation.__len__(), loss.item())
                )
            
            self.validationIter += 1
        
        self.validationAccuracy = self.calculateAccuracy("Validation")
        
    def test(self):
        self.classifier.eval()
        self.truePredicted = 0
        for c, batch in enumerate(self.dataLoaderTest):
            inputImage, gtLabel = batch["inputImage"].to(self.args["devices"]), batch["label"].to(self.args["devices"])
            
            predictedLabel = self.classifier(inputImage)
            
            loss = self.lossFunction(predictedLabel, gtLabel)
            
            self.compareLabels(predictedLabel, gtLabel)
            
            if (c % (int(self.dataLoaderTest.__len__() * self.args["logInterval"])) == 0) and (c != 0):                
                print(
                    "[Test] - [Epoch: %s / %s] - [Iter: %s / %s] - [Loss: %s]"
                    % (self.epoch, self.args["numEpochs"], c, self.dataLoaderTest.__len__(), loss.item())
                )
            
            self.testIter += 1
        
        self.testAccuracy = self.calculateAccuracy("Test")
            
    def train(self):
        self.classifier.train()
        self.truePredicted = 0
        for c, batch in enumerate(self.dataLoaderTrain):
            inputImage, gtLabel = batch["inputImage"].to(self.args["devices"]), batch["label"].to(self.args["devices"])
            
            self.optimizer.zero_grad()
            
            predictedLabel = self.classifier(inputImage)
            
            loss = self.lossFunction(predictedLabel, gtLabel)
            
            self.compareLabels(predictedLabel, gtLabel)
            
            loss.backward()
            self.optimizer.step()
            
            if (c % (int(self.dataLoaderTrain.__len__() * self.args["logInterval"])) == 0) and (c != 0):
                print(
                    "[Train] - [Epoch: %s / %s] - [Iter: %s / %s] - [Loss: %s]"
                    % (self.epoch, self.args["numEpochs"], c, self.dataLoaderTrain.__len__(), loss.item())
                )

            self.trainIter += 1
        
        self.trainAccuracy = self.calculateAccuracy("Train")
                
    def main(self):
        for self.epoch in range(self.args["numEpochs"]):
            self.train()
            self.test()
            self.validate()
            self.checkSave()