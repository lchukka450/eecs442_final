# affordance aware clip based representations of objects (ACRO)
import torch.nn as nn
import torch

class UnaryClassifier(nn.Module):
    def __init__(self, embed_size=512, hidden_size=256, num_classes=1):
        super(UnaryClassifier, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.unary_classifier = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        output = self.unary_classifier(x)

        if self.num_classes == 1:
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, dim=1)

        return output

class BinaryClassifier(nn.Module):
    def __init__(self, embed_size=512, hidden_size=256, num_classes=1):
        super(BinaryClassifier, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.binary_classifier = nn.Sequential(
            nn.Linear(embed_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.flatten(x, start_dim=1)
        output = self.binary_classifier(x)

        if self.num_classes == 1:
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, dim=1)

        return output

class AffordanceEncoder(nn.Module):
    def __init__(self,
                 input_size=512,
                 hidden_sizes=(1024, 1024),
                 embed_size=512):

        super(AffordanceEncoder, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.embed_size = embed_size

        self.enc = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], embed_size)
        )
    
    def forward(self, x):
        output = self.enc(x)
        return output
#Affordance-Aware CLIP-based Representation of Objects
class AcroModel(nn.Module):
    def __init__(self,
                 pretrained_encoder,
                 pretrained_emb_size=1000,
                 affordance_hidden_sizes=(1024, 1024),
                 affordance_emb_size=512,
                 clf_hidden_size=256,
                 num_materials=22,
    ):

        super(AcroModel, self).__init__()
        
        self.pretrained_encoder = pretrained_encoder
        self.pretrained_emb_size = pretrained_emb_size

        self.affordance_hidden_sizes = affordance_hidden_sizes
        self.affordance_embed_size = affordance_emb_size
        self.affordance_encoder = AffordanceEncoder(pretrained_emb_size, affordance_hidden_sizes, affordance_emb_size)

        self.clf_hidden_size = clf_hidden_size
        self.num_materials = num_materials

        self.liquid_clf = UnaryClassifier(affordance_emb_size, clf_hidden_size, 2)
        self.sealed_clf = UnaryClassifier(affordance_emb_size, clf_hidden_size, 2)
        self.material_clf = UnaryClassifier(affordance_emb_size, clf_hidden_size, num_materials)
        self.transparent_clf = UnaryClassifier(affordance_emb_size, clf_hidden_size, 2)

        self.deform_clf = BinaryClassifier(affordance_emb_size, clf_hidden_size, 2)
        self.fragility_clf = BinaryClassifier(affordance_emb_size, clf_hidden_size, 2)
        self.mass_clf = BinaryClassifier(affordance_emb_size, clf_hidden_size, 2)
    
    def forward(self, images):
        """
        images: (batch_size, 2, 3, 224, 224)
        """
        image_features = self.pretrained_encoder(images.squeeze(0)) # collapse pairs
        
        affordance_features = self.affordance_encoder(image_features)
        affordance_features = affordance_features.unsqueeze(0)

        # liquid_pred = self.liquid_clf(affordance_features)
        # sealed_pred = self.sealed_clf(affordance_features)
        # material_pred = self.material_clf(affordance_features)
        # transparent_pred = self.transparent_clf(affordance_features)

        deform_pred = self.deform_clf(affordance_features)
        fragility_pred = self.fragility_clf(affordance_features)
        mass_pred = self.mass_clf(affordance_features)

        return {
            # "liquid": liquid_pred,
            # "sealed": sealed_pred,
            # "material": material_pred,
            # "transparent": transparent_pred,
            "deform": deform_pred,
            "fragility": fragility_pred,
            "mass": mass_pred
        }