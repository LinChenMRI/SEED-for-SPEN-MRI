%% SEED reconstruction for SPEN MRI for SPEN MRI
% Lin Chen
% Xiamen University
% Email: chenlin@stu.xmu.edu.cn
% Aug. 5, 2015
% If you use this code, please cite the following papers:
% [1] L. Chen, J. Li, M. Zhang, S.H. Cai, T. Zhang, C.B. Cai, Z. Chen, Super-resolved enhancing and edge deghosting (SEED) for spatiotemporally encoded single-shot MRI, Med. Image Anal., 23 (2015) 1-14.

clear all; close all;clear classes; clc
addpath(strcat(pwd,'/Toolbox'));
%%
% * Create the time stamp*
date_now=datestr(now);
date_hour=num2str(hour(date_now));
date_minute=num2str(minute(date_now));
date_second=num2str(second(date_now));
time=['_',date_hour,'_',date_minute,'_',date_second];

%%
param.fft_number = 256; % The digital resolution of SR image

% * Input the directory of the FID file* 
param.fid_dir = 'Data'; % The directory of FID file
Procpar = readprocpar(param.fid_dir);   % load the experiment parameters

%% Get the experiment parameters
% * Some common paramters* 
param.gama = 42.58e-6;
param.segment = 1;  % The number of segments used in the experiment

% * Extract the paramters which are needed in the reconstruction process* 
param.Lpe = Procpar.lpe.Values*1e-2; % The FOV of phase encoded dimension
param.pw = Procpar.p1_bw.Values; % The bandwidth of the chirp pulse
param.Texc = Procpar.pw1.Values; % The duration of the chirp pulse
param.nphase = Procpar.nphase.Values; % The number of points in the phase encoded dimension
param.nread = Procpar.nread.Values/2; % The number of points in the frequency encoded dimension
param.chirp = 1; % 1 for 90 chirp and 2 for 180 chirp. 
param.sign = 1; % choose 1 or -1 according to the sweep direction of chirp

% * Calculate the other paramters* 
param.Gexc = param.sign*param.pw/(param.gama*param.Lpe);
param.gama = param.gama*2*pi; % translate the gama into rid form
%%
% *Load the original data in the SPEN sampling domain*
[RE,IM,NP,NB,NT,HDR]=load_fid(param.fid_dir,param.segment); % load fid data,choose the wanting segment
fid=RE+1i*IM;   % translate into complex form
fid=fid(1+param.nread*2:end,:); % discard the front useless data
zerofillnum=(param.fft_number-param.nread)/2;% the number of zero-filling points 

for n = 1:1:NT
    fid_temp = reshape(fid(:,n),param.nread,param.nphase).'; % Translate into 2D
    fid_temp(2:2:end,:) =  fliplr(fid_temp(2:2:end,:)); % Rearrange the even lines
    fid_temp_zerofill = [zeros(param.nphase,zerofillnum),fid_temp,zeros(param.nphase,zerofillnum)]; % Zero-fill the fid
    fid_temp_fftshift  = fftshift(fid_temp_zerofill,2);
    st_temp_fftshift = fft(fid_temp_fftshift,[],2);
    st_temp = fftshift(st_temp_fftshift,2);
    param.st(:,:,n) = st_temp;
end
figure(1);imshow(abs(param.st),[]);title('Blurred image');drawnow

%% *Create the directory for saving the results*
param.savedatadir = [param.fid_dir,'\result'];
if exist(param.savedatadir,'file')==0
    mkdir(param.savedatadir);
end

%% Super-resolved reconstruction
% param.SPP_position = -param.Lpe/2+(1:1:param.nphase)*param.Lpe/param.nphase;
param.samplingtrajectory = [0,ones(1,param.nphase-1)];
param.SPP_position = -param.Lpe/2+cumsum(param.samplingtrajectory)*param.Lpe/param.nphase;    % the position of stationary phase point
param.additional_term = param.gama*param.Gexc*param.Texc/(2*param.Lpe);
param.reconstruction_poistion = linspace(-param.Lpe/2,param.Lpe/2,param.fft_number);

for n=1:1:param.nphase
    param.alpha=param.additional_term*(param.reconstruction_poistion.^2-2*param.reconstruction_poistion*param.SPP_position(n));
    param.P_matrix(n,:)=exp(1i*param.alpha);
end
[U,S,V] = svd(param.P_matrix);
param.P_matrix = U*eye(size(S))*V';

param.undersamplefactor=param.pw*param.Texc/param.nphase;
param.reconstruction_poistion=linspace(-param.Lpe/2,param.Lpe/2,param.fft_number);
param.a=-param.gama*param.Gexc*param.Texc/(2*param.Lpe);
param.b=param.gama*param.Gexc*param.Texc/2;
param.deltay=param.Lpe/param.undersamplefactor;
param.keymatrix=diag(ones(1,param.fft_number));
for number=1:1:round(param.undersamplefactor-1)
    param.deltay=param.Lpe/param.undersamplefactor*number;
    param.deltaphase_left=-(param.additional_term*param.reconstruction_poistion.^2 - param.additional_term*(param.reconstruction_poistion-param.deltay).^2) ;
    param.deltaphase_right=-(param.additional_term*param.reconstruction_poistion.^2 - param.additional_term*(param.reconstruction_poistion+param.deltay).^2) ;
    param.aliasingnum=round(param.fft_number*(1-number/param.undersamplefactor))+1;
    right=[ones(1,param.aliasingnum),zeros(1,(param.fft_number-param.aliasingnum))];
    left=[zeros(1,(param.fft_number-param.aliasingnum)),ones(1,param.aliasingnum)];
    param.deltaroh_left=diag(exp(1i*param.deltaphase_left))*circshift(diag(left),[0,param.aliasingnum]);
    param.deltaroh_right=diag(exp(1i*param.deltaphase_right))*circshift(diag(right),[0,-param.aliasingnum]);
    param.keymatrix=param.keymatrix+param.deltaroh_left+param.deltaroh_right;
end

FSR = p1FSR(param);
result_without_CS = FSR'*param.st;
figure(2),imshow(abs(result_without_CS),[]);title('Super-resovled result');
imwrite(abs(result_without_CS)/max(abs(result_without_CS(:))),[param.savedatadir,'\result_without_CS.tiff'],'tiff')
%% Edge deghosting
CSparam=init;
CSparam.imsize=size(result_without_CS);
CSparam.BW=zeros(size(result_without_CS));
CSparam.BW(1,:)=result_without_CS(1,:);
CSparam.BW(2:end,:)=result_without_CS(2:end,:)-result_without_CS(1:end-1,:);
CSparam.BW_original=CSparam.BW;
tolerance=round(param.fft_number/param.undersamplefactor/2);
for n=1:1:size(CSparam.BW_original,2)
    CSparam.BW_original_fft(:,n)=fftshift(fft(fftshift(CSparam.BW_original(:,n))));
end
CSparam.BW_original_fft(param.fft_number/2-tolerance:param.fft_number/2+tolerance,:)=0;
for n=1:1:size(CSparam.BW_original,2)
    CSparam.BW_original_fft(:,n)=flipud(fftshift(fft(fftshift(CSparam.BW_original_fft(:,n)))));
end
CSparam.BW=CSparam.BW_original_fft;
gaussianFilter = fspecial('gaussian', [3, 3], 10);
CSparam.BW = imfilter(abs(CSparam.BW), gaussianFilter, 'circular', 'conv');
CSparam.BW = abs(CSparam.BW);
CSparam.BW=CSparam.BW/max(CSparam.BW(:))*0.8;
figure(3);imshow(abs(CSparam.BW),[]);title('Extract the prior knowledge of edge ghosts');drawnow

CSparam.FT=FSR;
CSparam.data=param.st/norm(param.st);
CSparam.Itnlim=20;
CSparam.TV=TVOP;
CSparam.XFM=1;
% CSparam.XFM=Wavelet('Daubechies',6,1);
CSparam.xfmWeight=1e-4;
CSparam.TVWeight=1e-4;
x=zeros([param.fft_number,param.fft_number]);

for n=1:1:5
    x=fnlCg(x,CSparam);
    figure(100), imshow(abs(x),[]), drawnow
end

result_with_CS = x;
figure(4),imshow(abs(result_with_CS),[]);title('Edge deghosting result');
imwrite(abs(result_with_CS)/max(abs(result_with_CS(:))),[param.savedatadir,'\result_with_CS.tiff'],'tiff')