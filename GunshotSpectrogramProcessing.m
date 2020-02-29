%%Compile Music 
clear all; close all; clc

mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Gavin_WeaponSounds\M4';
D=dir(fullfile(mainpath,'*.wav'));

%%  Construct Spectrogram images of music files
 numSeconds=7;
    for j=1:numel(D)
         thisfile = fullfile(mainpath,D(j).name);
         [gunshotfile,fs] = audioread(fullfile(mainpath,D(j).name));
         shortgunshot = gunshotfile(1:(numSeconds*fs),:);
         resampGunshot = [];
         for i=1:(floor(length(shortgunshot)/2))
             resampGunshot = [resampGunshot shortgunshot(i*2)];
         end
         nclip = length(resampGunshot);
         L = length(resampGunshot)/fs;
         tp2=linspace(0,L,nclip+1); 
         tp=tp2(1:nclip);
         if rem(length(resampGunshot),2) == 0
             kp=(2*pi/L)*[0:nclip/2-1 -nclip/2:-1];
             kps=fftshift(kp);
         else
             kp=(2*pi/L)*[0:nclip/2 -nclip/2:-1];
             kps=fftshift(kp);
         end      
%          figure(13)
%          plot((1:length(resampGunshot))/fs,resampGunshot); 
%          xlabel('Time [sec]'); 
%          ylabel('Amplitude');
%          title('Gunshot'); 
%          drawnow 
%          p8 = audioplayer(resampGunshot,fs);
%          playblocking(p8);           
     width=1214;
    step=zeros(1,length(tp));
    mask=ones(1,2*width+1);
    Shotstept_spec=[]; 
    tslide=(width+1):800:(length(step)-width);
    for z=1:length(tslide)
         step=zeros(1,length(tp));
         step(tslide(z)-width:1:tslide(z)+width)=mask;
         Shotstep=step.*resampGunshot; 
         Shotstept=fft(Shotstep); 
         Shotstept_spec=[Shotstept_spec; 
         abs(fftshift(Shotstept))];     
%          figure(1)
%          subplot(3,1,1), plot(tp,resampGunshot,'k',tp,step,'r')
%           xlabel('Time [sec]'); 
%           ylabel('Amplitude'); 
%           subplot(3,1,2), plot(tp,Shotstep,'k')
%           xlabel('Time [sec]'); 
%           ylabel('Amplitude'); 
%           subplot(3,1,3), plot(kps/(2*pi),abs(fftshift(Shotstept))/max(abs(Shotstept))) 
%           xlabel('Frequency [Hz]'); 
%           ylabel('|FFT(v)|')
%           axis([-1200 1200 0 1])
%           drawnow
    end

tslidep=linspace(0,L,length(tslide));
f=figure('visible', false);
pcolor(tslidep,kps/(2*pi),Shotstept_spec.'), 
shading interp 
set(gca,'XTick',[],'YTick',[],'Ylim',[0 2000],'Fontsize',[14]) 
colormap(hsv)
newFileName = split(thisfile,".");
filename = string(newFileName(1,1));
newFileNameChar = filename + '.jpg';
print(newFileNameChar,'-djpeg'); 
close(f)         
         
filescomplete = j
    end
      
    
    %% Reduce White Space on JPEGs
    
 mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Gavin_WeaponSounds\M4';
D=dir(fullfile(mainpath,'*.jpg'));   
    figure(8)
    for j=1:numel(D)
        thisfile = fullfile(mainpath,D(j).name);
        SpectroImg = imread(fullfile(mainpath,D(j).name));
        SpectroGray = double(rgb2gray(SpectroImg));
        SpectroCrop = SpectroGray(70:586,120:785);
        imshow(uint8(SpectroCrop))
        pause(.2)
        newFileName = split(thisfile,".");
        filename = string(newFileName(1,1));
        newFileNameChar = filename + '_cropped.jpg';
        imwrite(uint8(SpectroCrop),newFileNameChar); 
    end
    
    
    
    
    
    
    
    
    
    