Search.setIndex({docnames:["api","auralflow","auralflow.customs","auralflow.customs.init_model","auralflow.customs.setup_model","auralflow.datasets","auralflow.datasets.AudioDataset","auralflow.datasets.AudioFolder","auralflow.datasets.create_audio_dataset","auralflow.datasets.create_audio_folder","auralflow.datasets.load_stems","auralflow.datasets.verify_dataset","auralflow.losses","auralflow.losses.ComponentLoss","auralflow.losses.KLDivergenceLoss","auralflow.losses.SISDRLoss","auralflow.losses.component_loss","auralflow.losses.kl_div_loss","auralflow.losses.si_sdr_loss","auralflow.model.SeparationModel","auralflow.models","auralflow.models.SpectrogramMaskModel","auralflow.models.SpectrogramNetLSTM","auralflow.models.SpectrogramNetSimple","auralflow.models.SpectrogramNetVAE","auralflow.trainer","auralflow.trainer.run_training","auralflow.trainer.run_validation","auralflow.transforms","auralflow.transforms.AudioTransform","auralflow.transforms.trim_audio","auralflow.utils","auralflow.visualizer","contents","generated/auralflow.customs","generated/auralflow.datasets","generated/auralflow.losses","generated/auralflow.models","generated/auralflow.trainer","generated/auralflow.transforms","generated/auralflow.utils","generated/auralflow.visualizer","index","installation","modules","notes","params","quickstart","theory","usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api.rst","auralflow.rst","auralflow.customs.rst","auralflow.customs.init_model.rst","auralflow.customs.setup_model.rst","auralflow.datasets.rst","auralflow.datasets.AudioDataset.rst","auralflow.datasets.AudioFolder.rst","auralflow.datasets.create_audio_dataset.rst","auralflow.datasets.create_audio_folder.rst","auralflow.datasets.load_stems.rst","auralflow.datasets.verify_dataset.rst","auralflow.losses.rst","auralflow.losses.ComponentLoss.rst","auralflow.losses.KLDivergenceLoss.rst","auralflow.losses.SISDRLoss.rst","auralflow.losses.component_loss.rst","auralflow.losses.kl_div_loss.rst","auralflow.losses.si_sdr_loss.rst","auralflow.model.SeparationModel.rst","auralflow.models.rst","auralflow.models.SpectrogramMaskModel.rst","auralflow.models.SpectrogramNetLSTM.rst","auralflow.models.SpectrogramNetSimple.rst","auralflow.models.SpectrogramNetVAE.rst","auralflow.trainer.rst","auralflow.trainer.run_training.rst","auralflow.trainer.run_validation.rst","auralflow.transforms.rst","auralflow.transforms.AudioTransform.rst","auralflow.transforms.trim_audio.rst","auralflow.utils.rst","auralflow.visualizer.rst","contents.rst","generated/auralflow.customs.rst","generated/auralflow.datasets.rst","generated/auralflow.losses.rst","generated/auralflow.models.rst","generated/auralflow.trainer.rst","generated/auralflow.transforms.rst","generated/auralflow.utils.rst","generated/auralflow.visualizer.rst","index.rst","installation.rst","modules.rst","notes.rst","params.rst","quickstart.rst","theory.rst","usage.rst"],objects:{"":[[1,0,0,"-","auralflow"]],"auralflow.customs":[[3,1,1,"","init_model"],[4,1,1,"","setup_model"]],"auralflow.datasets":[[6,2,1,"","AudioDataset"],[7,2,1,"","AudioFolder"],[8,1,1,"","create_audio_dataset"],[9,1,1,"","create_audio_folder"],[10,1,1,"","load_stems"],[11,1,1,"","verify_dataset"]],"auralflow.datasets.AudioDataset":[[6,3,1,"","__getitem__"]],"auralflow.datasets.AudioFolder":[[7,3,1,"","__getitem__"],[7,3,1,"","__iter__"],[7,3,1,"","generate_sample"],[7,3,1,"","split"]],"auralflow.losses":[[13,2,1,"","ComponentLoss"],[14,2,1,"","KLDivergenceLoss"],[12,2,1,"","L1Loss"],[12,2,1,"","L2Loss"],[12,2,1,"","L2MaskLoss"],[12,2,1,"","RMSELoss"],[15,2,1,"","SISDRLoss"],[16,1,1,"","component_loss"],[12,1,1,"","get_evaluation_metrics"],[17,1,1,"","kl_div_loss"],[12,1,1,"","rmse_loss"],[18,1,1,"","si_sdr_loss"]],"auralflow.losses.ComponentLoss":[[13,3,1,"","forward"]],"auralflow.losses.KLDivergenceLoss":[[14,3,1,"","forward"]],"auralflow.losses.L1Loss":[[12,3,1,"","forward"]],"auralflow.losses.L2Loss":[[12,3,1,"","forward"]],"auralflow.losses.L2MaskLoss":[[12,3,1,"","forward"]],"auralflow.losses.RMSELoss":[[12,3,1,"","forward"]],"auralflow.losses.SISDRLoss":[[15,3,1,"","forward"]],"auralflow.models":[[20,2,1,"","SeparationModel"],[21,2,1,"","SpectrogramMaskModel"],[22,2,1,"","SpectrogramNetLSTM"],[23,2,1,"","SpectrogramNetSimple"],[24,2,1,"","SpectrogramNetVAE"]],"auralflow.models.SeparationModel":[[20,3,1,"","backward"],[20,3,1,"","compute_loss"],[20,3,1,"","eval"],[20,3,1,"","forward"],[20,3,1,"","load_grad_scaler"],[20,3,1,"","load_model"],[20,3,1,"","load_optim"],[20,3,1,"","load_scheduler"],[20,3,1,"","optimizer_step"],[20,3,1,"","save"],[20,3,1,"","save_grad_scaler"],[20,3,1,"","save_model"],[20,3,1,"","save_optim"],[20,3,1,"","save_scheduler"],[20,3,1,"","scheduler_step"],[20,3,1,"","separate"],[20,3,1,"","set_data"],[20,3,1,"","test"],[20,3,1,"","train"]],"auralflow.models.SpectrogramMaskModel":[[21,3,1,"","backward"],[21,3,1,"","compute_loss"],[21,3,1,"","forward"],[21,3,1,"","optimizer_step"],[21,3,1,"","scheduler_step"],[21,3,1,"","separate"],[21,3,1,"","set_data"]],"auralflow.models.SpectrogramNetLSTM":[[22,3,1,"","forward"],[22,3,1,"","split_lstm_parameters"]],"auralflow.models.SpectrogramNetSimple":[[23,3,1,"","forward"]],"auralflow.models.SpectrogramNetVAE":[[24,3,1,"","forward"]],"auralflow.separate":[[1,1,1,"","separate_audio"]],"auralflow.test":[[1,1,1,"","main"]],"auralflow.train":[[1,1,1,"","main"]],"auralflow.trainer":[[26,1,1,"","run_training"],[27,1,1,"","run_validation"]],"auralflow.transforms":[[29,2,1,"","AudioTransform"],[30,1,1,"","trim_audio"]],"auralflow.transforms.AudioTransform":[[29,3,1,"","audio_to_mel"],[29,3,1,"","pad_audio"],[29,3,1,"","to_audio"],[29,3,1,"","to_decibel"],[29,3,1,"","to_mel_scale"],[29,3,1,"","to_spectrogram"]],"auralflow.utils":[[31,1,1,"","copy_config_template"],[31,1,1,"","load_config"],[31,1,1,"","load_object"],[31,1,1,"","save_config"],[31,1,1,"","save_object"]],"auralflow.visualizer":[[32,2,1,"","ProgressBar"],[32,2,1,"","Visualizer"],[32,1,1,"","config_visualizer"]],"auralflow.visualizer.Visualizer":[[32,3,1,"","embed_audio"],[32,3,1,"","save_figure"],[32,3,1,"","test_model"],[32,3,1,"","visualize"],[32,3,1,"","visualize_gradient"],[32,3,1,"","visualize_spectrogram"],[32,3,1,"","visualize_waveform"]],auralflow:[[34,0,0,"-","customs"],[35,0,0,"-","datasets"],[36,0,0,"-","losses"],[37,0,0,"-","models"],[1,0,0,"-","separate"],[1,0,0,"-","test"],[1,0,0,"-","train"],[38,0,0,"-","trainer"],[39,0,0,"-","transforms"],[40,0,0,"-","utils"],[41,0,0,"-","visualizer"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method"},terms:{"0":[5,7,12,13,14,16,17,20,22,23,47,48,49],"0008":47,"06":12,"1":[5,7,8,12,13,14,16,17,18,20,22,23,32,46,48],"10":[12,15,18,28,29,32,48],"100":[5,8,10,47],"1000":[5,8],"10000":[5,6,8],"1024":[20,22],"11":43,"16":[12,16,17,18,20,23],"16000":12,"173":[12,16,17],"1e":12,"2":[5,6,7,8,9,12,13,14,15,16,17,18,20,23,46],"20":[1,28,29],"200":1,"22050":[5,8,9],"256":[12,17],"2c":[12,13,16],"3":[5,6,7,12,13,16,20,22,24,32,43,48],"30":1,"306":48,"32":47,"3c":[12,13,16],"4":49,"4096":47,"44100":[1,5,6,7,8,9,10,28,29,32],"5":[20,23,47],"512":[12,16,17],"7":43,"7858":48,"8":[12,16,47],"8192":47,"88200":[12,18],"9":32,"\u03f5":48,"abstract":[19,20],"class":[5,6,7,8,9,12,13,14,15,16,17,18,19,20,21,22,23,24,28,29,32,35,36,37,39,41,47],"default":[5,6,7,8,9,10,11,12,14,20,22,23,28,29,31,49],"final":[20,23],"float":[5,7,12,13,16,17,18,19,20,21,23,46],"function":[12,17,19,20,34,35,36,38,39,40,41,46,48,49],"import":[2,3,4,5,8,9],"int":[1,5,6,7,8,9,10,12,19,20,22,23,28,29,31,32,46],"new":[2,3,4,49],"return":[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,28,29,30,31,32],"short":[33,45],"true":[2,4,5,7,8,9,10,12,15,18,19,20,28,29,32],"while":[12,48],A:[5,10,11,19,20,33,42,45,49],As:48,By:48,For:[12,15,18],If:[2,4,12,13,16,19,20,28,29,48],In:[12,14,17,48],The:[5,7,11,12,13,14,16,17,48,49],There:48,To:[2,3,48,49],With:42,_2:[12,13,15,16,18],_:[32,48],__:32,__getitem__:[6,7],__iter__:7,a_i:48,abc:20,absolut:48,access:[20,21],accord:[2,4],achiev:49,activ:[20,23,46,47],adam:48,addit:[20,22,24,48],afterward:12,against:[12,13,16],ai:47,aka:[20,23,46],algorithm:48,all:[12,19,20,28,30,48],allud:48,alpha:[12,13,16,20,23,46],also:[12,14],although:[12,48],altogeth:48,among:[19,20],amount:[12,13,16],amplitud:48,an:[5,7,8,9,12,15,18,19,20,28,29,32,42,48,49],angl:48,ani:[5,11,19,20,49],api:33,appli:[20,21,48],applic:[2,4],appropri:[19,20],approx:48,approxim:48,ar:[5,7,48],architectur:[20,22,24,42,46,47],arg:[12,13,20,22,24],argmin_:48,argument:[12,13,16,20,22,24,49],arrai:[28,30,42],attach:31,attenu:[12,13,16],attribut:[2,4,12,13,14,15,19,20],audio:[1,5,6,7,8,9,10,11,12,15,18,19,20,21,28,29,30,32,33,35,42,46,47,49],audio_format:[5,7,9],audio_tensor:[28,30],audio_to_mel:[28,29],audiodataset:[5,8],audiofold:[5,9],audiotransform:28,auralflow:[33,47,49],automat:[2,4,19,20,46],avail:47,avoid:48,axi:[20,22,23],back:48,backend:[5,7,9],background:[5,10,12,16,42],backpropag:[19,20,21],backward:[19,20,21],balanc:[12,13,16],bar:[32,48],base:[1,2,3,4,5,6,7,12,13,14,15,20,21,22,23,24,25,26,27,28,29,32,37,42,46,47],basic:33,bass:[5,10,11,42],batch:[12,19,20,21,23,46,47,48],batch_loss:[12,13,14,15,19,20],been:[2,4,48],befor:[19,20,46],begin:48,belong:[5,8,9,10,11,48],below:49,best:[2,4,19,20],beta:[12,13,16,46],between:[12,16],bin:[20,23,28,29,46],bool:[5,8,9,10,19,20,21,23,28,29,32,46],bottleneck:[20,22,24],build:47,c:[20,23,48],calcul:[12,13,14,15,16,17,18,19,20,48],call:[12,19,20,48,49],callabl:[19,20],callback:[25,26],can:[12,14,15,17,18,42,48,49],cannot:[2,4,5,9,10,31],canon:48,care:12,carefulli:48,cdot:48,certain:49,channel:[5,7,20,23,46,48],check:[2,4],checkpoint:[2,4,19,20,31,46,49],checkpoint_path:[19,20],choic:48,choos:48,chunk:[5,6,7,8,9,46],chunk_siz:[5,6,8,9],citeseerx:48,clear:48,clip:49,clone:31,close:[12,14,17],code:42,codomain:48,collect:[5,10],combin:[12,16,17],command:33,common:48,complex:[28,29,48],complex_spec:[28,29],complic:48,compon:[12,13,16],component_loss:[12,13,46],componentloss:12,comput:[12,17,20,21,28,29],compute_loss:[19,20,21],condit:[20,24],confi:49,config:[19,20,31,32,33,47],config_data:[2,3,4],config_filepath:1,config_visu:32,configur:[1,2,3,4,19,20,21,31,33],consist:[5,10],constant:[20,23,46],construct:[12,14],constructor:[20,22,24],contain:[5,11,49],content:[3,49],conveni:[28,29,49],convent:48,convert:[28,29],copi:31,copy_config_templ:[2,3,4,31],correct:48,correspond:6,cover:49,cpu:[28,29,46],creat:[2,3,4,5,6,8,9,20,23,46,49],create_audio_dataset:5,create_audio_fold:5,criterion:[2,4,12,14,19,20,46,47,48],csv:[1,49],current:[19,20,21,43,49],custom:[33,42,47],customiz:[42,49],cv:49,d_:[12,14,17],d_kl:[12,14,17],data:[2,3,5,6,7,10,12,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,46,48],data_path:[5,8,9],dataload:[5,7,25,26,27],dataset:[19,20,33,42,49],dataset_param:[2,3,19,20],dataset_path:[5,7,8,9,10],decibel:[28,29],decod:[20,24],decreas:[19,20],deep:[20,21,22,23,42,47],defin:[12,13,14,15,16,17,18,48],definit:33,depend:[12,13,16],describ:48,design:[5,7,42],desir:49,detail:48,deviat:[12,14,17],devic:[19,20,23,28,29],dft:48,dict:[2,3,19,20,21,31,32],dict_kei:[2,3],dictionari:[2,3,4,31],differ:[12,16],dim:[20,22],dimens:48,dimension:48,directli:[5,6,7,10,12,13,14,15,18,19,20],directori:[2,3,4,5,7,8,9,10,11,31,49],dirti:48,disclaim:48,discret:48,displai:49,distort:[12,18],distribut:[12,14,17,20,24],div:[12,17],dive:[33,42,45],diverg:[12,17],divis:[28,29],doe:[19,20,21],doi:48,domain:[12,18,20,21,28,29,48],download:[42,48],dropout:[20,23,46],dropout_p:[20,23,49],drum:[5,10,11,42],durat:[1,5,6,7,8,9,10,28,29,30],dure:32,e_:48,each:[5,6,8,9,10,11,48,49],ear:48,earli:[19,20,21,46],earlier:48,edit:49,editor:33,edu:48,effici:[5,7,42],element:[12,14,17,48],embed_audio:32,emploi:48,encapsul:49,encod:[20,24],end:[5,11],ensur:48,entir:42,entri:48,ep:12,epoch:[19,20,46,47],error:48,essenti:49,estim:[12,15,16,17,18,19,20,21,22,23,46,48],estimate_spec:[12,17],eta:[12,13],eval:[12,19,20],evalu:[2,12,15,18,19,20,34,42,49],everi:[5,11,12],exampl:[2,3,4,5,8,9,12,16,17,18,49],exist:[19,20,48],exp:48,expect:31,express:[12,14,17],extend:48,extrem:49,f:[12,13,16,20,23,48],factor:[46,48],fals:[20,23,32],far:49,fascin:47,featur:[20,22,23,42,49],feed:[20,22,24,48],few:48,fft:[20,23,28,29,46,47],figur:32,file:[1,2,3,4,5,7,8,9,10,11,31,35,47,49],filenam:[1,32],filenotfounderror:[2,4],filepath:31,fill:[2,4,19,20],filter:[12,13,16,46],filterbank:[20,23,46,48],find:48,first:[28,29,49],flag:[19,20],floattensor:[12,16,17,18,19,20,22,23,24],fly:[5,7],folder:[2,3,4,5,7,9,11,19,20,49],follow:[5,11,19,20,28,29],foral:48,form:[12,14,17],format:[5,7,9],former:12,forward:[12,13,14,15,19,20,21,22,23,24],found:[2,4],four:49,fourier:[33,45],frac:[12,13,14,15,16,17,18,28,29],frame:[20,23],framework:32,frequenc:[20,22,23,28,29,48],from:[5,7,8,9,10,19,20,22,32,33,46,47,48],full:[5,6,8,10,11,42,49],g:49,gaussian:[20,24],gener:[5,7,20,24],generate_sampl:[5,7],get:[6,12,16,17,18],get_evaluation_metr:12,getcwd:[2,3,4,5,8,9],given:[1,2,3,19,20,31,48],global:31,global_step:[19,20,31,32],go:48,good:48,gpu:46,grad:[19,20],grad_scal:[19,20],gradient:[2,4,19,20,21,32,47],griffin:48,ground:[5,6,8,9,10,12,16,18],ha:48,hadamard:48,hasattr:[2,4],hat:[12,13,15,16,18,48],have:[2,4,48],helper:[5,7,8,9],here:49,heta:[12,13],hidden:[20,22,46],hidden_channel:[20,23],hidden_s:[20,22],highli:42,histori:[19,20],hook:12,hop:[28,29,46,47],hop_length:[28,29],how:48,howev:48,http:48,huge:48,i:[5,12,14,17,35,48],id:49,idx:[6,7],ignor:12,imag:[32,49],imagefold:[5,7],imaginari:48,implement:[19,20,37],improv:[19,20,21],includ:[2,4,5,11,42],inde:48,index:[6,7,42],individu:[5,11],infer:[19,20,32],inform:48,init_model:[2,4],initi:[20,23,32,46,49],input:[19,20,23,46,47,48],input_axi:[20,22],insid:49,instanc:[2,3,12,19,20],instanti:[2,4,19,20,34],instead:[5,7,12,13,14,15],instrument:42,integr:[12,36],interfac:[19,20],intern:[2,4,19,20,21],introduct:48,invari:[12,18],invers:48,invok:49,involv:48,ioerror:[5,9,10,11,31],is_best:[19,20],ist:48,item:[12,16,17,18],iter:7,iterabledataset:[5,7],its:[2,4,12,16,17,18,19,20,28,29,48,49],itself:[5,7],j:48,jame:47,json:[31,49],k:[12,13,16,48],kei:[2,3,49],keyword:[20,22,24,49],kl:[12,14,17],kl_div_loss:[12,14],kl_term:[12,17],kldivergenceloss:12,knowledg:48,known:48,kwarg:[20,22,24],l1:[12,14],l1_loss:[12,17],l1_term:[12,17],l1loss:12,l2:12,l2loss:12,l2maskloss:12,l:48,l_:[12,13,15,16,18],label:[5,6,8,9,10,11,19,20,32],last:48,lastli:42,latent:[20,24],latest:43,latter:12,layer:[20,22,23,46],leak:46,leak_factor:[20,23],leaki:[20,23],leaky_relu:46,learn:[2,4,19,20,21,23,42,46],learnabl:46,length:[28,29,46,47],let:48,lib:32,librari:32,lightweight:42,like:49,lim:48,line:33,linear:48,list:[5,6,7,8,9,10,11,19,20,22,28,30,46,49],listen:32,literatur:48,ln:[12,14,17],load:[2,3,4,5,6,7,8,9,10,19,20,31,46,49],load_config:[2,3,4,31],load_grad_scal:[19,20],load_model:[19,20],load_object:31,load_optim:[19,20],load_schedul:[19,20],load_stem:5,log:[28,29,32,46,49],log_:[12,15,18,28,29],look:49,loop:[25,26,27],loss:[19,20,21,33,42,46,48],loss_fn:[12,14],loss_val:[12,16,17,18],lpha:[12,13],lr:[19,20,21,46,47],lstm:[20,22,24],m_:[12,13,16,48],magnitud:[28,29,48],mai:48,main:[1,49],make:[28,29,48],manag:43,map:[1,5,10,12,48],mask:[12,13,14,15,16,20,21,22,23,46,47,48],mask_act_fn:[20,23],mask_activ:49,mask_model:[12,13,14],match:[28,30],math:[28,29],mathbb:48,matplotlib:32,max:[5,8,10,19,20,28,29,46,47],max_lr_step:[19,20],max_num_track:[5,8,10],max_track:1,mean:[12,14,16,17,18,48],meet:[5,11],mel:[28,29],memori:[5,6,7,10,46],mention:48,messag:46,method:[5,7,8,9,19,20,21,22,23,24,48],metric:[1,12,15,18],mini:48,minim:[42,48],misc:[2,4],miss:[5,11],mix:[2,4,19,20,21,46,47],mixtur:[5,6,7,10,11,12,16,19,20,21,32,48],mixture_audio:32,ml:42,mode:[2,4,19,20,32],model:[1,2,3,4,12,13,14,15,17,25,26,27,31,32,33,34,42,47,48],model_dir:[2,3,4],model_nam:[19,20],model_param:[2,3,19,20],modern:49,modifi:33,modul:[4,13,14,15,19,23,36,37,47],mono:[5,8,9,10,20,23,46],more:[42,49],moreov:[42,48,49],most:48,mse:48,mu:[12,14,17],much:48,mul:[28,29],multipl:[20,23,28,29,48],musdb18:47,music:[42,48],must:[5,11,19,20,48],my_model:[2,3,4,47],n:[12,13,14,16,17,20,23],name:[7,19,20,46,48],nativ:42,ndarrai:[5,10,28,30],necessari:[48,49],need:[12,47],neq:48,net:[20,24],network:[2,4,12,16,42,48],neural:42,new_valu:49,next:[48,49],nn:[2,4,12,13,14,15,17,19,20,23,36,37],nois:[12,13,16,48],non:[20,22],none:[1,5,7,8,9,10,11,12,13,14,15,19,20,21,23,25,26,27,31,32],normal:[12,14,17,20,23,28,29,46,47,48],normalize_input:[20,23],normalize_output:[20,23],note:[12,13,33],num:[46,47],num_batch:12,num_channel:[5,7,20,23],num_chunk:[5,6,8],num_fft:[28,29],num_fft_bin:[20,23],num_fram:[20,23],num_imag:32,number:[5,6,7,8,10,12,14,17,19,20,22,23,28,29,46],numpi:[5,10,28,30],o:[5,35],obj_nam:31,object:[5,7,19,20,25,26,28,29,31,32,35,48,49],odot:[12,13,16,48],off:[5,7],offer:42,often:48,one:[5,10,12,19,20,28,30,49],ones:[12,17],onli:[12,14,17,19,20],open:49,optim:[2,4,12,15,18,19,20,48],optimizer_step:[19,20,21],option:[5,7,8,9,10,11,20,21,23,49],order:[5,10],ordereddict:[5,10],organ:49,os:[2,3,4,5,8,9],oserror:[2,4,31],other:[5,10,11,12,13,16],otherwis:[2,4,19,20],our:48,out:47,output:[12,16,19,20,23,46],overfit:48,overlap:48,overridden:12,p:[12,14,17,20,24,46,48],packag:[32,42,43,49],package_nam:43,pad:[1,28,29],pad_audio:[28,29],pair:[5,7,48,49],param:31,paramet:[2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,33,48],parameter:48,parameter_group:49,parameter_nam:49,part:48,pass:[12,49],path:[5,7,8,9,10,11,19,20,31,32,49],pathlib:32,patienc:46,pdf:48,perform:[12,20,21,49],phase:48,pin:46,pip:43,play_audio:32,plot:49,pool:46,port:42,posit:[20,22,24],precis:[2,4,19,20,46,47],predica:48,presenc:49,pretrain:42,pretti:48,previou:[19,20],prior:48,probabl:[20,23,46],process:[19,20,21,46],processor:42,product:48,progress:[32,47],progressbar:32,proj_:[12,15,18],psu:48,pth:31,purpos:48,put:48,py:32,pypi:43,python3:32,python:[32,33,42,43],pytorch:[5,7,19,20,37,42,47],q:[12,14,17],qualiti:[12,13,16,42],quick:48,quickstart:33,quit:48,r:[12,13,16,48],r_:[12,13,16],r_f:[12,13,16],rac:[12,13],rais:[2,4,5,9,10,11,31],rand:[12,16,17,18],randomli:[5,7],rate:[2,4,5,7,8,9,10,19,20,28,29,46],rather:[19,20,48],ratio:[5,7],raw:[28,29],read:[5,8,9,10,31],recent:48,recip:12,recommend:33,reconstruct:[12,17],recurrent_depth:[20,22],reduc:[19,20,21,46],reduct:[19,20,46],regist:12,relat:49,relu:[20,23,49],rep1:48,rep:48,replac:[5,7,49],represent:[28,29,48],request:[20,22],requir:[5,11,19,20,48,49],resampl:[5,6,8,9,10,46],resample_r:1,residu:[12,13,16],respect:[6,7],result:[32,42],rm:48,rmse:12,rmse_loss:12,rmseloss:12,root:[5,7],rule:[5,11],run:[1,12,19,20,25,26,27,32,49],run_train:25,run_valid:25,s:[2,4,5,7,19,20,21,22,31,48,49],s_:48,sampl:[5,7,8,9,10,20,24,28,29,46,48],sample_length:[5,7],sample_r:[5,6,7,8,9,10,28,29,32],save:[1,19,20,31,32,49],save_audio:32,save_config:31,save_dir:[2,3,4,31,32],save_figur:32,save_filepath:[1,31],save_freq:32,save_grad_scal:[19,20],save_imag:32,save_model:[19,20],save_object:31,save_optim:[19,20],save_schedul:[19,20],scalar:[12,16,17,18,19,20],scale:[12,18,28,29],scaler:[2,4,19,20],schedul:[2,4,19,20],scheduler_step:[19,20,21],schoolboi:47,score:12,scratch:49,script:[1,42],sdr:[12,15,18],seamless:42,second:[5,6,8,9,46,48],see:49,seen:49,segment:48,send:32,separ:[2,4,12,13,16,19,20,21,22,25,26,27,31,37,42,47,48,49],separate_audio:1,separationmodel:[1,2,3,4,12,20,21,25,26,27,31,36],set:[2,4,5,7,12,13,14,15,19,20,21,25,26,27,33],set_data:[19,20,21],setup:[2,34],setup_model:2,sgd:48,shallow:[33,42,45],shape:[20,23],share:[19,20],shell:49,shine:48,shortest:[28,30],should:[12,19,20],si:[12,15,18],si_sdr:47,si_sdr_loss:[12,15],sigma:[12,14,17],sigmoid:[20,23,47],signal:[5,6,8,9,10,12,15,18,20,21,28,29,33],silenc:[19,20],silent:[12,46],silent_checkpoint:[19,20],similar:[5,7],simpl:[48,49],simpler:48,simplest:49,simpli:[48,49],sinc:12,singl:[1,12,15,18,49],sisdrloss:12,site:32,size:[20,23,28,29,46,47],small:48,so:49,soft:[12,16,20,23,46,48],solv:48,some:[2,4,48,49],sound:48,soundfil:[5,7,9],sourc:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,42,46,47,48],specifi:49,spectrogram:[12,13,14,15,16,17,20,21,23,24,28,29,32,33,47,49],spectrogrammaskmodel:[12,13,14,15,20],spectrogramnetlstm:[20,24],spectrogramnetsimpl:[20,22,49],spectrogramnetva:[12,14,17,20,47],speed:46,split:[5,7,8,9,10,42],split_lstm_paramet:[20,22],sport:42,squar:48,sr:[1,12],stack:[20,22],standard:[12,14,17],start:49,state:[2,4,19,20,31,49],std:[12,17,19,20,32],stem:[1,5,10],step:[31,46,47],stereo:[20,23,46],stft:[46,48],stop:[19,20,21,46],stop_pati:[19,20],store:[5,7,12,13,14,15,28,29,31,32],str:[1,5,6,7,8,9,10,11,12,14,19,20,23,28,29,31,32,46],structur:[5,11,49],subclass:[12,19,20],subdirectori:[5,11],subroutin:[28,29],subset:[5,7,8,9,10,11],subset_path:[5,11],sum_:[12,14,17],summarywrit:32,suppress:46,t:[20,23,48],take:12,target:[5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,32,46,48],target_audio:32,target_label:[19,20],target_spec:[12,17],task:48,tau:48,templat:49,tempor:[20,23],tenor:[28,29],tensor:[1,5,6,7,12,14,16,17,18,19,20,21,28,29,30,32],tensorboard:[32,49],term:[12,13,14,16,17],test:[2,5,7,8,9,11,19,20,34,49],test_dataset:[5,8,9],test_model:32,text:[12,15,18,33],them:12,theori:[33,42,45],theta:[12,16,48],thi:[12,48,49],third:[12,13,16],three:[12,13,16],time:[19,20,22,23,28,29,33,45],titl:[5,11],to_audio:[28,29],to_db:[28,29],to_decibel:[28,29],to_mel_scal:[28,29],to_spectrogram:[28,29],tool:[28,29,42],toolkit:42,torch:[1,5,6,7,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32],torchaudio:[5,7,9],toy_dataset:[5,8,9],tqdm:32,track:[1,5,6,7,8,9,10,11,12,15,18,19,20,42,46],train:[2,4,5,7,8,9,10,11,19,20,25,26,32,33,34,42,47,48,49],train_dataload:[25,26],train_loss:[19,20],trainer:[33,42],training_mod:[19,20],training_param:[2,3,19,20],trainingcallback:[25,26],transfer:46,transform:[20,21,33,42,45],transport:48,trick:48,trim:[28,30],trim_audio:28,trivial:48,truth:[5,6,8,9,10,12,16,18],tupl:[5,6,7,20,22],two:[12,13,16],type:[2,3,4,5,6,7,8,9,10,12,16,17,18,19,20,28,29,30,31,46,48],typic:[12,15,18],u:[20,24],uncertainti:48,under:31,underli:[2,4,19,20,37],understand:48,union:[19,20,28,30,32],unit:[12,13,16],unknown:48,up:[2,4,46],updat:[19,20,21],upgrad:43,us:[2,3,12,13,14,15,16,17,18,19,20,22,23,28,29,42,46,47,48,49],usag:33,use_amp:[19,20],use_pad:[28,29],util:[2,3,4,5,6,7,25,26,27,32,33,42],vae:[20,24],val:[5,11,20,21],val_dataload:[25,26,27],val_loss:[19,20],val_split:[5,7],valid:[5,7,11,19,20,25,26,27,49],valu:[12,13,16,17,18,19,20,28,29,48,49],vanilla:[20,23],variabl:[19,20,24],variou:[2,4],ve:48,verifi:[5,9,10,11],verify_dataset:5,versatil:42,version:[2,4,32,43],via:[43,49],view:47,view_gradi:32,view_spectrogram:32,view_waveform:32,viewdoc:48,visual:[19,20,33,42,47,49],visualize_gradi:32,visualize_spectrogram:32,visualize_waveform:32,visualizer_param:[2,3,19,20],vocal:[5,7,10,11,42],wa:48,wai:[48,49],wait:[19,20],wav:[5,7,9,11,47],waveform:[32,47,49],we:[48,49],weigh:[12,13,16],weight:[12,13,16,32],well:48,when:49,where:[5,10,12,13,14,15,16,17,18,20,23,31,48],whether:[20,22,23],which:[2,4,48,49],window:[28,29,46,47,48],window_s:[28,29],wise:[12,19,20,21,48],within:[5,11,12,31,33],without:[5,7,19,20,47,48],work:48,worker:[46,47],workflow:42,wrap:32,wrapper:[12,13,14,15,20,21,28,29,32],write:[2,3,4,47],writer:32,x:[12,13,16,20,24,28,29,48],y:[12,15,18,28,29,48],y_:[12,13,16,48],y_k:48,yield:[7,48],you:49,your_model:49,z:[20,24],zero:[12,17,28,29]},titles:["Python API","auralflow package","auralflow.customs","auralflow.customs.init_model","auralflow.customs.setup_model","auralflow.datasets","auralflow.datasets.AudioDataset","auralflow.datasets.AudioFolder","auralflow.datasets.create_audio_dataset","auralflow.datasets.create_audio_folder","auralflow.datasets.load_stems","auralflow.datasets.verify_dataset","auralflow.losses","auralflow.losses.ComponentLoss","auralflow.losses.KLDivergenceLoss","auralflow.losses.SISDRLoss","auralflow.losses.component_loss","auralflow.losses.kl_div_loss","auralflow.losses.si_sdr_loss","auralflow.models.SeparationModel","auralflow.models","auralflow.models.SpectrogramMaskModel","auralflow.models.SpectrogramNetLSTM","auralflow.models.SpectrogramNetSimple","auralflow.models.SpectrogramNetVAE","auralflow.trainer","auralflow.trainer.run_training","auralflow.trainer.run_validation","auralflow.transforms","auralflow.transforms.AudioTransform","auralflow.transforms.trim_audio","auralflow.utils","auralflow.visualizer","&lt;no title&gt;","auralflow.customs","auralflow.datasets","auralflow.losses","auralflow.models","auralflow.trainer","auralflow.transforms","auralflow.utils","auralflow.visualizer","Auralflow Documentation","Installation","auralflow","Notes","Parameters","Quickstart","A Shallow Dive into Theory","Basic Usage"],titleterms:{"short":48,A:48,api:[0,42],audio:48,audiodataset:6,audiofold:7,audiotransform:29,auralflow:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44],basic:49,command:49,component_loss:16,componentloss:13,config:49,configur:49,content:[1,2,5,12,20,25,28,31,32],create_audio_dataset:8,create_audio_fold:9,custom:[2,3,4,34],dataset:[5,6,7,8,9,10,11,35,46],definit:48,dive:48,document:42,editor:49,fourier:48,from:49,indic:42,init_model:3,instal:43,kl_div_loss:17,kldivergenceloss:14,line:49,load_stem:10,loss:[12,13,14,15,16,17,18,36],model:[19,20,21,22,23,24,37,46,49],modifi:49,modul:[1,2,5,12,20,25,28,31,32],note:[42,45],packag:1,paramet:[46,49],python:0,quickstart:[42,47],recommend:49,requir:43,run_train:26,run_valid:27,separ:1,separationmodel:19,set:49,setup_model:4,shallow:48,si_sdr_loss:18,signal:48,sisdrloss:15,spectrogram:48,spectrogrammaskmodel:21,spectrogramnetlstm:22,spectrogramnetsimpl:23,spectrogramnetva:24,subpackag:1,system:43,tabl:42,test:1,text:49,theori:48,time:48,train:[1,46],trainer:[25,26,27,38],transform:[28,29,30,39,48],trim_audio:30,updat:43,usag:49,util:[31,40],verify_dataset:11,visual:[32,41],within:49}})