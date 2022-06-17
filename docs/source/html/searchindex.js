Search.setIndex({docnames:["api","auralflow","auralflow.datasets","auralflow.datasets.AudioDataset","auralflow.datasets.AudioFolder","auralflow.datasets.create_audio_dataset","auralflow.datasets.create_audio_folder","auralflow.datasets.load_stems","auralflow.datasets.verify_dataset","auralflow.losses","auralflow.losses.component_loss","auralflow.losses.kl_div_loss","auralflow.losses.si_sdr_loss","auralflow.models","auralflow.models.SpectrogramMaskModel","auralflow.models.SpectrogramNetLSTM","auralflow.models.SpectrogramNetSimple","auralflow.models.SpectrogramNetVAE","auralflow.trainer","auralflow.utils","auralflow.visualizer","contents","generated/auralflow.datasets","generated/auralflow.losses","generated/auralflow.models","generated/auralflow.trainer","generated/auralflow.utils","generated/auralflow.visualizer","index","modules","params","quickstart","usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api.rst","auralflow.rst","auralflow.datasets.rst","auralflow.datasets.AudioDataset.rst","auralflow.datasets.AudioFolder.rst","auralflow.datasets.create_audio_dataset.rst","auralflow.datasets.create_audio_folder.rst","auralflow.datasets.load_stems.rst","auralflow.datasets.verify_dataset.rst","auralflow.losses.rst","auralflow.losses.component_loss.rst","auralflow.losses.kl_div_loss.rst","auralflow.losses.si_sdr_loss.rst","auralflow.models.rst","auralflow.models.SpectrogramMaskModel.rst","auralflow.models.SpectrogramNetLSTM.rst","auralflow.models.SpectrogramNetSimple.rst","auralflow.models.SpectrogramNetVAE.rst","auralflow.trainer.rst","auralflow.utils.rst","auralflow.visualizer.rst","contents.rst","generated/auralflow.datasets.rst","generated/auralflow.losses.rst","generated/auralflow.models.rst","generated/auralflow.trainer.rst","generated/auralflow.utils.rst","generated/auralflow.visualizer.rst","index.rst","modules.rst","params.rst","quickstart.rst","usage.rst"],objects:{"":[[1,0,0,"-","auralflow"]],"auralflow.datasets":[[3,1,1,"","AudioDataset"],[4,1,1,"","AudioFolder"],[5,3,1,"","create_audio_dataset"],[6,3,1,"","create_audio_folder"],[7,3,1,"","load_stems"],[8,3,1,"","verify_dataset"]],"auralflow.datasets.AudioDataset":[[3,2,1,"","__getitem__"]],"auralflow.datasets.AudioFolder":[[4,2,1,"","__getitem__"],[4,2,1,"","__iter__"],[4,2,1,"","split"]],"auralflow.losses":[[9,1,1,"","KLDivergenceLoss"],[9,1,1,"","L1Loss"],[9,1,1,"","L2Loss"],[9,1,1,"","L2MaskLoss"],[9,1,1,"","RMSELoss"],[9,1,1,"","SIDRLoss"],[9,1,1,"","WeightedComponentLoss"],[10,3,1,"","component_loss"],[9,3,1,"","get_model_criterion"],[11,3,1,"","kl_div_loss"],[12,3,1,"","si_sdr_loss"]],"auralflow.losses.KLDivergenceLoss":[[9,2,1,"","forward"]],"auralflow.losses.L1Loss":[[9,2,1,"","forward"]],"auralflow.losses.L2Loss":[[9,2,1,"","forward"]],"auralflow.losses.L2MaskLoss":[[9,2,1,"","forward"]],"auralflow.losses.RMSELoss":[[9,2,1,"","forward"]],"auralflow.losses.SIDRLoss":[[9,2,1,"","forward"]],"auralflow.losses.WeightedComponentLoss":[[9,2,1,"","forward"]],"auralflow.models":[[13,1,1,"","SeparationModel"],[14,1,1,"","SpectrogramMaskModel"],[15,1,1,"","SpectrogramNetLSTM"],[16,1,1,"","SpectrogramNetSimple"],[17,1,1,"","SpectrogramNetVAE"],[13,3,1,"","create_model"],[13,3,1,"","setup_model"]],"auralflow.models.SeparationModel":[[13,2,1,"","backward"],[13,2,1,"","compute_loss"],[13,2,1,"","eval"],[13,2,1,"","forward"],[13,2,1,"","load_grad_scaler"],[13,2,1,"","load_model"],[13,2,1,"","load_optim"],[13,2,1,"","load_scheduler"],[13,2,1,"","optimizer_step"],[13,2,1,"","save"],[13,2,1,"","save_grad_scaler"],[13,2,1,"","save_model"],[13,2,1,"","save_optim"],[13,2,1,"","save_scheduler"],[13,2,1,"","scheduler_step"],[13,2,1,"","separate"],[13,2,1,"","set_data"],[13,2,1,"","test"],[13,2,1,"","train"]],"auralflow.models.SpectrogramMaskModel":[[14,2,1,"","backward"],[14,2,1,"","compute_loss"],[14,2,1,"","forward"],[14,2,1,"","optimizer_step"],[14,2,1,"","scheduler_step"],[14,2,1,"","separate"],[14,2,1,"","set_data"]],"auralflow.models.SpectrogramNetLSTM":[[15,2,1,"","forward"],[15,2,1,"","split_lstm_parameters"]],"auralflow.models.SpectrogramNetSimple":[[16,2,1,"","forward"]],"auralflow.models.SpectrogramNetVAE":[[17,2,1,"","forward"]],"auralflow.separate":[[1,3,1,"","separate_audio"]],"auralflow.test":[[1,3,1,"","main"]],"auralflow.train":[[1,3,1,"","main"]],"auralflow.trainer":[[18,3,1,"","run_training"],[18,3,1,"","run_validation"]],"auralflow.utils":[[19,1,1,"","AudioTransform"],[19,3,1,"","load_config"],[19,3,1,"","load_object"],[19,3,1,"","pull_config_template"],[19,3,1,"","save_config"],[19,3,1,"","save_object"],[19,3,1,"","trim_audio"]],"auralflow.utils.AudioTransform":[[19,2,1,"","audio_to_mel"],[19,2,1,"","pad_audio"],[19,2,1,"","to_audio"],[19,2,1,"","to_decibel"],[19,2,1,"","to_mel_scale"],[19,2,1,"","to_spectrogram"]],"auralflow.visualizer":[[20,1,1,"","ProgressBar"],[20,1,1,"","Visualizer"],[20,3,1,"","config_visualizer"]],"auralflow.visualizer.Visualizer":[[20,2,1,"","embed_audio"],[20,2,1,"","save_figure"],[20,2,1,"","test_model"],[20,2,1,"","visualize"],[20,2,1,"","visualize_gradient"],[20,2,1,"","visualize_spectrogram"],[20,2,1,"","visualize_waveform"]],auralflow:[[22,0,0,"-","datasets"],[23,0,0,"-","losses"],[24,0,0,"-","models"],[1,0,0,"-","separate"],[1,0,0,"-","test"],[1,0,0,"-","train"],[25,0,0,"-","trainer"],[26,0,0,"-","utils"],[27,0,0,"-","visualizer"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[2,4,9,10,11,13,15,16,31,32],"0008":31,"1":[2,4,9,10,11,13,15,16,20,30],"10":[9,12,20],"100":[2,5,7,31],"10000":[2,3,5],"1024":[13,15],"16":[9,10,13,16],"173":[9,10],"2":[2,3,4,5,6,9,10,11,12,13,16,30],"20":[1,19],"200":1,"2c":[9,10],"3":[2,3,4,9,10,13,15,17,20],"30":1,"32":31,"3c":[9,10],"4":32,"4096":31,"44100":[1,2,3,4,5,6,7,19,20],"5":[13,16,31],"512":[9,10],"8":[9,10,31],"8192":31,"9":20,"abstract":13,"class":[2,3,4,9,13,14,15,16,17,19,20,22,23,24,26,27,31],"default":[2,3,4,5,6,7,9,13,15,16,32],"final":[13,16],"float":[2,4,9,10,13,14,16,30],"function":[9,22,23,24,25,26,27,30,32],"import":[9,10],"int":[1,2,3,4,5,6,7,13,15,16,19,20,30],"new":[13,19,32],"return":[1,2,3,4,5,6,7,9,10,11,12,13,14,20],"static":19,"true":[2,4,5,6,7,9,12,13,19,20],"while":9,A:[2,7,8,32],For:[9,12],If:[9,10],In:[9,11],The:[2,4,8,9,10,11,32],To:32,With:28,_2:[9,10,12],_:20,__:20,__getitem__:[3,4],__iter__:4,abc:13,access:[13,14],accord:9,achiev:32,activ:[13,16,30,31],addit:[13,15,17],afterward:9,against:[9,10],ai:31,aka:[13,16,30],all:[9,13],alpha:[9,10,13,16,30],also:9,although:9,among:13,amount:[9,10],an:[2,4,5,6,9,12,13,19,20,28,32],ani:[2,8,32],api:21,appli:[13,14,19],ar:[2,4],architectur:[13,15,17,28,30,31],arg:[9,13,15,17],argument:[9,10,13,15,17,32],arrai:28,attach:19,attenu:[9,10],attribut:9,audio:[1,2,3,4,5,6,7,8,9,12,13,14,19,20,22,28,30,31,32],audio_format:[2,4,6],audio_tensor:19,audio_to_mel:19,audiodataset:[2,5],audiofold:[2,6],audiotransform:19,auralflow:[21,31,32],automat:30,avail:31,axi:[13,15,16],backend:[2,4,6],background:[2,7,9,10,28],backprop:[9,10],backpropag:[13,14],backward:[9,10,13,14],balanc:[9,10],bar:20,base:[1,2,3,4,9,13,14,15,16,17,18,19,20,28,30,31],basic:21,bass:[2,7,8,28],batch:[9,10,13,14,16,30,31],batch_loss:9,befor:30,belong:[2,5,6,7,8],below:32,beta:[9,10,30],between:[9,10,13],bin:[13,16,30],blank:13,bool:[2,5,6,7,13,14,16,19,20,30],bottleneck:[13,15,17],build:31,c:[13,16],calcul:[9,10],call:[9,13,32],callabl:9,callback:18,can:[9,11,12,28,32],cannot:[2,7],care:9,certain:32,channel:[2,4,13,16,30],checkpoint:[19,30,32],chunk:[2,3,4,5,6,30],chunk_siz:[2,3,5,6],clip:32,close:[9,11],code:28,collect:[2,7],combin:[9,10],command:21,complex:19,complex_spec:19,compon:[9,10],component_loss:[9,30],comput:[9,11,13,14,19],compute_loss:[13,14],condit:[13,17],confi:32,config:[9,13,19,20,21,31],config_filepath:[1,19],config_visu:20,configur:[1,9,13,14,19,21],consist:[2,7],constant:[13,16,30],construct:9,constructor:[13,15,17],contain:[2,8,32],content:32,conveni:[19,32],copi:19,correspond:3,cover:32,cpu:[19,30],creat:[2,3,5,6,13,16,30,32],create_audio_dataset:2,create_audio_fold:2,create_model:13,criterion:[9,30,31],csv:[1,32],current:[13,14,32],custom:[28,31],customiz:[28,32],cv:32,d_:[9,11],d_kl:[9,11],data:[2,3,4,7,9,10,13,14,15,16,17,18,19,30],dataload:[2,4,18],dataset:[21,28,32],dataset_path:[2,4,5,6,7],decibel:19,decod:[13,17],decreas:13,deep:[13,14,15,16,28,31],defin:[9,10,11,12],depend:[9,10],design:[2,4,28],desir:32,deviat:[9,11],devic:[13,16,19],dict:[9,13,14,19,20],differ:[9,10],dim:[13,15],directli:[2,3,4,7,9,12],directori:[2,4,5,6,7,8,19,32],displai:32,distort:[9,12],distribut:[9,11,13,17],diverg:[9,11],document:21,doe:[13,14],domain:[9,12,13,14,19],dropout:[13,16,30],dropout_p:[13,16,32],drum:[2,7,8,28],durat:[1,2,3,4,5,6,7],dure:20,each:[2,3,5,6,7,8,32],earli:[13,14,30],edit:32,editor:21,effici:[2,4,28],element:[9,11],embed_audio:20,encapsul:32,encod:[13,17],end:[2,8],entir:28,epoch:[30,31],essenti:32,estim:[9,10,12,13,14,15,16,30],eta:9,eval:13,evalu:[9,12,13,28,32],everi:[2,8,9],exampl:[9,10,32],exist:13,express:[9,11],extrem:32,f:[9,10,13,16],factor:30,fals:[13,16,20],far:32,fascin:31,featur:[13,15,16,28,32],feed:[13,15,17],fft:[13,16,30,31],figur:20,file:[1,2,4,5,6,7,8,9,19,22,31,32],filenam:[1,20],filepath:19,filter:[9,10,30],filterbank:[13,16,30],first:32,floattensor:[9,10,11,12,13,15,16,17],fly:[2,4],folder:[2,4,6,8,32],follow:[2,8],form:[9,11],format:[2,4,6],former:9,forward:[9,13,14,15,16,17],four:32,frac:[9,10,11,12],frame:[13,16,19],framework:20,freq:19,frequenc:[13,15,16],from:[2,4,5,6,7,13,15,19,20,21,30,31],full:[2,3,5,7,8,28,32],g:32,gaussian:[13,17],gener:[2,4,9,10,13,17],get:[3,9],get_model_criterion:9,given:[1,13,19],global_step:[13,19,20],go:28,gpu:30,grad:13,grad_scal:13,gradient:[13,14,20,31],ground:[2,3,5,6,7,9,10,12],hat:[9,10,12],have:19,helper:[2,5,6],here:32,heta:9,hidden:[13,15,30],hidden_channel:[13,16],hidden_s:[13,15],highli:28,hook:9,hop:[30,31],hop_length:19,i:[2,9,11,22],id:32,idx:[3,4],ignor:9,imag:[20,32],imagefold:[2,4],improv:[13,14],includ:[2,8,28],index:[3,4,28],individu:[2,8],infer:20,initi:[13,16,20,30,32],input:[13,16,19,30,31],input_axi:[13,15],insid:32,instanc:[9,13],instead:[2,4,9],instrument:28,interfac:13,intern:[13,14],invari:[9,12],invok:32,ioerror:[2,7,8],item:[9,10],iter:4,iterabledataset:[2,4],its:[9,11,13,19,32],itself:[2,4],jame:31,json:[19,32],k:[9,10],kei:32,keyword:[13,15,17,32],kl:[9,11],kl_div_loss:9,kldivergenceloss:9,kwarg:[13,15,17],l1:9,l1loss:9,l2:9,l2loss:9,l2maskloss:9,l_:[9,10,12],label:[2,3,5,6,7,8,20],lastli:28,latent:[13,17],latter:9,layer:[13,15,16,30],leak:30,leak_factor:[13,16],leaki:[13,16],leaky_relu:30,learn:[13,14,16,28,30],learnabl:30,length:[30,31],lib:20,librari:20,lightweight:28,like:32,line:21,list:[2,3,4,5,6,7,8,13,15,19,30,32],listen:20,ln:[9,11],load:[2,3,4,5,6,7,13,19,30,32],load_config:19,load_grad_scal:13,load_model:13,load_object:19,load_optim:13,load_schedul:13,load_stem:2,locat:19,log10:19,log:[19,20,30,32],log_:[9,12],look:32,loop:18,loss:[13,14,21,28,30],loss_fn:9,loss_val:[9,10],lpha:9,lr:[13,14,30,31],lstm:[13,15,17],m_:[9,10],magnitud:19,main:[1,32],manag:28,map:[1,2,7],mask:[9,10,13,14,15,16,30,31],mask_act_fn:[13,16],mask_activ:32,match:19,matplotlib:20,max:[2,5,7,19,30,31],max_num_track:[2,5,7],max_track:1,mean:[9,10,11,12],meet:[2,8],mel:19,memori:[2,3,4,7,30],messag:30,method:[2,5,6,13,14,15,16,17],metric:[1,9,12],minim:28,miss:[2,8],mix:[13,14,30,31],mixtur:[2,3,4,7,8,9,10,13,14,20],mixture_audio:20,ml:28,mode:[13,20],model:[1,9,11,18,19,20,21,28,31],model_wrapp:19,modern:32,modifi:21,modul:[16,31],mono:[2,5,6,7,13,16,30],more:[28,32],moreov:[28,32],mu:[9,11],multipl:[13,16,19],musdb18:31,music:28,must:[2,8],my_model:31,n:[9,10,11,13,16],name:[4,30],nativ:28,ndarrai:[2,7,19],necessari:[13,32],need:[9,31],net:[13,17],network:[9,10,28],neural:28,new_valu:32,next:32,nn:[9,13,16],nois:[9,10],non:[13,15],none:[1,2,4,7,8,9,13,14,16,18,19,20],normal:[9,11,13,16,19,30,31],normalize_input:[13,16],normalize_output:[13,16],note:9,num:[30,31],num_channel:[2,4,13,16],num_chunk:[2,3,5],num_fft:19,num_fft_bin:[13,16],num_fram:[13,16],num_imag:20,number:[2,3,4,5,7,9,11,13,15,16,19,30],numpi:[2,7,19],o:[2,22],obj_nam:19,object:[2,13,19,20,22,32],odot:[9,10],off:[2,4],offer:28,one:[2,7,9,13,32],onli:9,open:32,optim:[9,12,13],optimizer_step:[13,14],option:[2,4,7,13,14,16,32],order:[2,7],ordereddict:[2,7],organ:32,other:[2,7,8,9,10],out:31,output:[9,10,13,16,30],overridden:9,p:[9,11,13,17,30],packag:[20,28,32],pad:[1,19],pad_audio:19,pair:[4,32],paramet:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21],parameter_group:32,parameter_nam:32,pass:[9,32],path:[2,4,5,6,7,8,20,32],pathlib:20,patienc:30,perform:[9,13,14,32],pin:30,pip:28,play_audio:20,plot:32,pool:30,port:28,posit:[13,15,17],precis:[13,30,31],presenc:32,pretrain:28,previou:13,probabl:[13,16,30],process:[13,14,30],processor:28,progress:[20,31],progressbar:20,proj_:[9,12],pth:19,pull_config_templ:19,py:20,pypi:28,python3:20,python:20,pytorch:[2,4,28,31],q:[9,11],qualiti:[9,10,28],quickstart:21,r:[9,10],r_:[9,10],r_f:[9,10],rac:9,rais:[2,7,8],rand:[9,10],randomli:[2,4],rate:[2,4,5,6,7,13,30],ratio:[2,4],raw:19,read:[2,5,6,7],readi:28,recip:9,recommend:21,recurrent_depth:[13,15],reduc:[13,14,30],reduct:30,regist:9,relat:32,relu:[13,16,32],replac:[2,4,32],represent:19,request:[13,15],requir:[2,8,32],resampl:[2,3,5,6,7,30],resample_r:1,residu:[9,10],respect:[3,4],result:[20,28],rmse:9,rmseloss:9,root:[2,4],rule:[2,8],run:[1,9,18,20,32],run_train:18,run_valid:18,s:[2,4,13,14,15,19,32],sampl:[2,4,5,6,7,13,17,30],sample_length:[2,4],sample_r:[2,3,4,5,6,7,19,20],save:[1,13,19,20,32],save_audio:20,save_config:19,save_dir:[19,20],save_figur:20,save_filepath:[1,19],save_freq:20,save_grad_scal:13,save_imag:20,save_model:13,save_object:19,save_optim:13,save_schedul:13,scalar:[9,10],scale:[9,12,19],scaler:13,schedul:13,scheduler_step:[13,14],schoolboi:31,scratch:32,script:[1,28],sdr:[9,12],seamless:28,second:[2,3,5,6,30],see:32,seen:32,send:20,separ:[9,10,13,14,15,28,31,32],separate_audio:1,separationmodel:[1,13,14,18],set:[2,4,9,13,14,21],set_data:[13,14],setup_model:13,shape:[13,16],share:13,shell:32,should:9,si:[9,12],si_sdr:31,si_sdr_loss:9,sidrloss:9,sigma:[9,11],sigmoid:[13,16,31],signal:[2,3,5,6,7,9,12,13,14,19],silent:[9,30],similar:[2,4],simpl:32,simplest:32,simpli:32,sinc:9,singl:[1,9,12,32],site:20,size:[13,16,30,31],so:32,soft:[9,10,13,16,30],some:32,soundfil:[2,4,6],sourc:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,28,30,31],specifi:32,spectrogram:[9,10,13,14,16,17,19,20,31,32],spectrogrammaskmodel:[9,13],spectrogramnetlstm:[13,17],spectrogramnetsimpl:[13,15,32],spectrogramnetva:[9,13,31],speed:30,split:[2,4,5,6,7,28],split_lstm_paramet:[13,15],sport:28,sr:1,stack:[13,15],standard:[9,11],start:32,state:[13,19,32],std:20,stem:[1,2,7],step:[30,31],stereo:[13,16,30],stft:30,stop:[13,14,30],store:[2,4,9,19,20],str:[1,2,3,4,5,6,7,8,9,13,16,19,20,30],structur:[2,8,32],subclass:9,subdirectori:[2,8],subset:[2,4,5,6,7,8],subset_path:[2,8],sum_:[9,11],summarywrit:20,suppress:30,t:[13,16],take:9,target:[2,3,4,5,6,7,8,9,10,12,13,14,20,30],target_audio:20,templat:[19,32],tempor:[13,16],tensor:[1,3,4,9,10,11,12,13,14,19,20],tensorboard:[20,32],term:[9,10,11],test:[2,4,8,13,32],test_model:20,text:[9,12,21],them:9,theta:[9,10],thi:[9,32],third:[9,10],three:[9,10],time:[13,15,16,19],titl:[2,8],to_audio:19,to_db:19,to_decibel:19,to_mel_scal:19,to_spectrogram:19,tool:[19,28],toolkit:28,torch:[1,2,3,4,9,10,11,12,13,14,15,16,17,18,19,20],torchaudio:[2,4,6],tqdm:20,track:[1,2,3,4,5,6,7,8,9,12,13,28,30],train:[2,4,5,6,7,8,13,18,20,21,28,31,32],train_dataload:18,trainer:[21,28],trainingcallback:18,transfer:30,transform:[13,14,19],trim:19,trim_audio:19,truth:[2,3,5,6,7,9,10,12],tupl:[3,4,13,15],two:[9,10],type:[2,3,4,5,6,7,9,10,11,12,30],typic:[9,12],u:[13,17],under:19,union:[9,19,20],unit:[9,10],up:[13,30],updat:[13,14],us:[9,10,11,12,13,15,16,28,30,31,32],usag:21,use_pad:19,util:[2,3,4,18,20,21,28],vae:[13,17],val:[2,8,13,14],val_dataload:18,val_split:[2,4],valid:[2,4,8,18,32],valu:[9,10,19,32],vanilla:[13,16],variabl:[13,17],verifi:[2,7,8],verify_dataset:2,versatil:28,version:20,via:[28,32],view:31,view_gradi:20,view_spectrogram:20,view_waveform:20,visual:[21,28,31,32],visualize_gradi:20,visualize_spectrogram:20,visualize_waveform:20,vocal:[2,4,7,8,28],wai:32,wav:[2,4,6,8,31],waveform:[20,31,32],we:32,weigh:[9,10],weight:[9,10,20],weightedcomponentloss:9,when:32,where:[2,7,9,10,11,12,13,16],whether:[13,15,16],which:32,window:[30,31],window_s:19,wise:[13,14],within:[2,8,9,21],without:[2,4,13,31],worker:[30,31],workflow:28,wrap:20,wrapper:[9,13,14,19,20],write:31,writer:20,x:[9,10,13,17,19],y:[9,12,19],y_:[9,10],yield:4,you:32,your_model:32,z:[13,17],zero:19},titles:["API Documentation","auralflow package","auralflow.datasets","auralflow.datasets.AudioDataset","auralflow.datasets.AudioFolder","auralflow.datasets.create_audio_dataset","auralflow.datasets.create_audio_folder","auralflow.datasets.load_stems","auralflow.datasets.verify_dataset","auralflow.losses","auralflow.losses.component_loss","auralflow.losses.kl_div_loss","auralflow.losses.si_sdr_loss","auralflow.models","auralflow.models.SpectrogramMaskModel","auralflow.models.SpectrogramNetLSTM","auralflow.models.SpectrogramNetSimple","auralflow.models.SpectrogramNetVAE","auralflow.trainer","auralflow.utils","auralflow.visualizer","&lt;no title&gt;","auralflow.datasets","auralflow.losses","auralflow.models","auralflow.trainer","auralflow.utils","auralflow.visualizer","Auralflow Documentation","auralflow","Parameters","Quickstart","Basic Usage"],titleterms:{api:[0,28],audiodataset:3,audiofold:4,auralflow:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29],basic:32,command:32,component_loss:10,config:32,configur:32,content:[1,2,9,13,18,19,20],create_audio_dataset:5,create_audio_fold:6,dataset:[2,3,4,5,6,7,8,22,30],document:[0,28],editor:32,from:32,indic:28,instal:28,kl_div_loss:11,line:32,load_stem:7,loss:[9,10,11,12,23],model:[13,14,15,16,17,24,30,32],modifi:32,modul:[1,2,9,13,18,19,20],packag:1,paramet:[30,32],python:28,quickstart:31,recommend:32,separ:1,set:32,si_sdr_loss:12,spectrogrammaskmodel:14,spectrogramnetlstm:15,spectrogramnetsimpl:16,spectrogramnetva:17,subpackag:1,tabl:28,test:1,text:32,train:[1,30],trainer:[18,25],usag:32,util:[19,26],verify_dataset:8,visual:[20,27],within:32}})