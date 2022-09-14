from metaflow import FlowSpec, step, Parameter

from src.Trainer import Trainer

def getTrainer(args):
    trainer = Trainer(args=args)
    return trainer

class MetaFlowOptunaTraining(FlowSpec):
    backbone1 = Parameter("backbone1", default="resnet50", help="Backbone architecture for training", type=str)
    backbone2 = Parameter("backbone2", default="resnet34", help="Backbone architecture for training", type=str)
    lr = Parameter("lr", default=0.001, help="Learning rate", type=float)
    experimentName = Parameter("experimentName", default="GenderClassification", help="Experiment name for training", type=str)
    datasetName = Parameter("datasetName", default="GenderDataset", help="Name of the dataloader of training")
    projectKey = Parameter("neptuneProjectKey", default="AGC", help="Project key for connection to Neptune.ai", type=str)
    neptuneModelType = Parameter("neptuneModelType", default="AGC-GCLA", help="Model type for versioning to Neptune.ai / ['CEN-ECLA', 'CEN-GCLA']", type=str)
    optimizer = Parameter("optimizer", default="Adam", help="Which optimizer to use in training", type=str)    
    momentum = Parameter("momentum", default=0.9, help="Momentum of optimizer", type=float)
    
    # *Train
    numEpochs = Parameter("numEpochs", default=1, help="Number of epochs", type=int)
    accuracyThreshold = Parameter("accuracyThreshold", default=0.8, help="Limit of validation accuracy to decide save model or not", type=float)
    devices = Parameter("devices", default="cuda:0", help="CUDA or CPU", type=str)
    pretrained = Parameter("pretrained", default=True, help="Load ImageNet weight or not", type=bool)
    lossFunction = Parameter("lossFunction", default="BCEWithLogitsLoss", help="Loss function to use in training", type=str)
    logInterval = Parameter("logInterval", default=0.1, help="Milestone percentage of log metrics in one epoch", type=float)
    savePath = Parameter("savePath", default="Checkpoints", help="Path to save trained model", type=str)
        
    # *Data
    numberClass = Parameter("numberClass", default=2, help="Number of classes")
    imagePath = Parameter("imagePath", default="src/Data/Images", help="Folder path that include training images", type=str)
    informationCSV = Parameter("informationCSV", default="src/Data/TrainGenderInformation.csv", help="Information CSV path for dataloader", type=str)
    validationImagePath = Parameter("validationImagePath", default="src/Data/ImagesValidation", help="Folder path that include validation images", type=str)
    validationInformationCSV = Parameter("validationInformationCSV", default="src/Data/ValidationGenderInformation.csv", help="Information CSV path for validation dataloader", type=str)
    seed = Parameter("seed", default=42, help="Random seed", type=int)
    batchSize = Parameter("batchSize", default=2, help="Input batch size for training", type=int)
    imageSize = Parameter("imageSize", default=64, help="Input image size for training", type=int)
    numWorkers = Parameter("numWorkers", default=4, help="Number of worker for dataloader", type=int)
    testSize = Parameter("testSize", default=0.05, help="Percentage of test images number", type=float)
    
    @step
    def start(self):
        self.next(self.trainmodel1, self.trainmodel2)
        
    @step
    def trainmodel1(self):
        self.trainer = getTrainer({
            "backbone": self.backbone1,
            "lr": self.lr,
            "experimentName": self.experimentName,
            "datasetName": self.datasetName,
            "projectKey": self.projectKey,
            "neptuneModelType": self.neptuneModelType,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "numEpochs": self.numEpochs,
            "accuracyThreshold": self.accuracyThreshold,
            "devices": self.devices,
            "pretrained": self.pretrained,
            "lossFunction": self.lossFunction,
            "logInterval": self.logInterval,
            "savePath": self.savePath,
            "numberClass": self.numberClass,
            "imagePath": self.imagePath,
            "informationCSV": self.informationCSV,
            "validationImagePath": self.validationImagePath,
            "validationInformationCSV": self.validationInformationCSV,
            "seed": self.seed,
            "batchSize": self.batchSize,
            "imageSize": self.imageSize,
            "numWorkers": self.numWorkers,
            "testSize": self.testSize
        })
        self.trainer.main()
        self.next(self.join)
        
    @step
    def trainmodel2(self):
        self.trainer = getTrainer({
            "backbone": self.backbone2,
            "lr": self.lr,
            "experimentName": self.experimentName,
            "datasetName": self.datasetName,
            "projectKey": self.projectKey,
            "neptuneModelType": self.neptuneModelType,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "numEpochs": self.numEpochs,
            "accuracyThreshold": self.accuracyThreshold,
            "devices": self.devices,
            "pretrained": self.pretrained,
            "lossFunction": self.lossFunction,
            "logInterval": self.logInterval,
            "savePath": self.savePath,
            "numberClass": self.numberClass,
            "imagePath": self.imagePath,
            "informationCSV": self.informationCSV,
            "validationImagePath": self.validationImagePath,
            "validationInformationCSV": self.validationInformationCSV,
            "seed": self.seed,
            "batchSize": self.batchSize,
            "imageSize": self.imageSize,
            "numWorkers": self.numWorkers,
            "testSize": self.testSize
        })
        self.trainer.main()
        self.next(self.join)
        
    @step
    def join(self, inputs):
        print("Train1 Validation Accuracy: ", inputs.trainmodel1.trainer.validationAccuracy)
        print("Train2 Validation Accuracy: ", inputs.trainmodel2.trainer.validationAccuracy)
        self.next(self.end)
        
    @step
    def end(self):
        print("Training completed!")
        
if __name__ == '__main__':
    MetaFlowOptunaTraining()