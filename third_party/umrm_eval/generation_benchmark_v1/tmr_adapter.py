"""Strict local-only official HumanML3D TMR inference. No fallback evaluator."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from setup_tmr import ASSETS, digest


def official_module(name, relative, package=False):
    path=ASSETS/'official'/relative
    entries=json.loads((ASSETS/'sources.json').read_text())
    if digest(path)!=entries[relative]['sha256']:
        raise ValueError('Official source hash changed: '+relative)
    if name in sys.modules:
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)] if package else None)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def preprocess(joints):
    """20Hz common22 y-up meters -> official retargeted 263D features (T-1)."""
    x=np.asarray(joints,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(22,3) or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('Expected finite T>=3,22,3 y-up joints at 20Hz')
    official_module('umrm_tmr_guofeats','src/guofeats/__init__.py',True)
    from umrm_tmr_guofeats.motion_representation import joints_to_guofeats
    features=joints_to_guofeats(x.copy()).astype(np.float32)
    if features.shape!=(len(x)-1,263) or not np.isfinite(features).all():
        raise ValueError('Invalid official TMR features')
    return features


def find_model():
    candidates=list((ASSETS/'pretrained').glob('**/tmr_humanml3d_guoh3dfeats/config.json'))
    if len(candidates)!=1:
        raise FileNotFoundError('Verified official TMR archive not installed; run setup_tmr.py --archive PATH. No substitute model is allowed.')
    return candidates[0].parent


class OfficialTMR:
    def __init__(self, text_model_dir, device='cpu'):
        folder=find_model()
        provenance=json.loads((ASSETS/'archive_provenance.json').read_text())
        for relative,expected in provenance['files'].items():
            if digest(ASSETS/'pretrained'/relative)!=expected:
                raise ValueError('Model archive file changed')
        cfg=json.loads((folder/'config.json').read_text())
        if cfg['data']['text_to_token_emb']['modelname']!='distilbert-base-uncased':
            raise ValueError('Unexpected text model')
        if not cfg['model']['vae']:
            raise ValueError('Expected official variational TMR; use mean token')
        actor=official_module('umrm_tmr_actor','src/model/actor.py')
        self.device=torch.device(device)
        for modality,nfeats in [('motion',263),('text',768)]:
            c=dict(cfg['model'][modality+'_encoder'])
            if c.pop('_target_')!='src.model.ACTORStyleEncoder' or c['nfeats']!=nfeats:
                raise ValueError('Incompatible TMR encoder config')
            module=actor.ACTORStyleEncoder(**c)
            state=torch.load(folder/'last_weights'/f'{modality}_encoder.pt',map_location='cpu',weights_only=True)
            module.load_state_dict(state,strict=True)
            setattr(self,modality+'_encoder',module.to(self.device).eval())
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer=AutoTokenizer.from_pretrained(str(text_model_dir),local_files_only=True)
        self.backbone=AutoModel.from_pretrained(str(text_model_dir),local_files_only=True).to(self.device).eval()
        if self.backbone.config.model_type!='distilbert' or self.backbone.config.hidden_size!=768:
            raise ValueError('Wrong text backbone')
        stats=ASSETS/'official/stats/humanml3d/guoh3dfeats'
        self.mean=torch.load(stats/'mean.pt',map_location=self.device,weights_only=True)
        self.std=torch.load(stats/'std.pt',map_location=self.device,weights_only=True)

    @torch.inference_mode()
    def encode_text(self,texts):
        inputs=self.tokenizer(texts,return_tensors='pt',padding=True).to(self.device)
        tokens=self.backbone(**inputs).last_hidden_state
        encoded=self.text_encoder({'x':tokens,'mask':inputs.attention_mask.bool()})[:,0]
        return torch.nn.functional.normalize(encoded,dim=-1).cpu().numpy()

    @torch.inference_mode()
    def encode_features(self,features):
        x=torch.as_tensor(features,dtype=torch.float32,device=self.device)
        x=(x-self.mean)/(self.std+1e-12)
        encoded=self.motion_encoder({'x':x[None],'mask':torch.ones((1,len(x)),dtype=torch.bool,device=self.device)})[:,0]
        return torch.nn.functional.normalize(encoded,dim=-1).cpu().numpy()[0]
