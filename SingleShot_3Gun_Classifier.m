%% Compiles elements of test data into individual matrices
clear all; close all; clc
     mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Single Shots\AK47\1k spectrograms';

    D=dir(fullfile(mainpath,'*cropped.jpg'));  
Band2Samp=[];
Band2AveSamp=[];

    for j=1:numel(D)
           song = imread(fullfile(mainpath,D(j).name));
         blurspectro = double(imresize(song, [64,64]));
%          imshow(blurspectro)
         avespectro = double(blurspectro) - mean(double(blurspectro));
         skinnyspectro =reshape(blurspectro,1,4096);
         skinnyavespectro = reshape(avespectro,1,4096);
         Band2AveSamp = [skinnyavespectro' Band2AveSamp];
         Band2Samp = [skinnyspectro' Band2Samp];
    end
         mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Single Shots\FAL\1k spectrograms';
    D=dir(fullfile(mainpath,'*cropped.jpg'));  
Band3Samp=[];
Band3AveSamp=[];

    for j=1:numel(D)
           song = imread(fullfile(mainpath,D(j).name));
         blurspectro = double(imresize(song, [64,64]));
%          imshow(blurspectro)
         avespectro = double(blurspectro) - mean(double(blurspectro));
         skinnyspectro =reshape(blurspectro,1,4096);
         skinnyavespectro = reshape(avespectro,1,4096);
         Band3AveSamp = [skinnyavespectro' Band3AveSamp];
         Band3Samp = [skinnyspectro' Band3Samp];
    end   
         mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Single Shots\M4\1k spectrograms';
        D=dir(fullfile(mainpath,'*cropped.jpg'));  
Band4Samp=[];
Band4AveSamp=[];

    for j=1:numel(D)
           song = imread(fullfile(mainpath,D(j).name));
         blurspectro = double(imresize(song, [64,64]));
%          imshow(blurspectro)
         avespectro = double(blurspectro) - mean(double(blurspectro));
         skinnyspectro =reshape(blurspectro,1,4096);
         skinnyavespectro = reshape(avespectro,1,4096);
         Band4AveSamp = [skinnyavespectro' Band4AveSamp];
         Band4Samp = [skinnyspectro' Band4Samp];
    end
    
    %% Wavelet Transform
    
%   band1_wave = dc_wavelet(Band1AveSamp);
    band2_wave = dc_wavelet(Band2AveSamp);
    band3_wave = dc_wavelet(Band3AveSamp);
    band4_wave = dc_wavelet(Band4AveSamp);
    
    [U,S,V]=svd([band2_wave,band3_wave,band4_wave],0);
    %% U Analysis
    
    figure(2)
for j=1:12
  subplot(3,4,j) 
  ut1 = reshape(U(:,j),32,32); 
  ut2 = ut1(32:-1:1,:); 
  pcolor(ut2), colormap(jet)
  set(gca,'Xtick',[],'Ytick',[])
  colorbar
end

%% Analyze S

figure(3)
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14]) 
axis([0 96 0 1.6*10^4])
xlabel('Singular Values')
ylabel('\sigma')
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14])
axis([0 96 0 1.5*10^4])
xlabel('Singular Values')
ylabel('\sigma')

%Singular value assessment
sig=diag(S);
energy1=sig(1)/sum(sig)
energy2=sig(2)/sum(sig) 
%eng1perc=sum(sig(1:2))/sum(sig)
eng10perc=sum(sig(1:10))/sum(sig)
eng25perc=sum(sig(1:24))/sum(sig)
eng50perc=sum(sig(1:49))/sum(sig)
eng=sum(sig(1:80))/sum(sig)
    
%% Analyze V
k=0;
figure(4)
for j=1:3
    k=k+1;
  subplot(3,3,k) 
  plot(1:32,V(1:32,j),'ko-') 
  k=k+1;
  subplot(3,3,k) 
  plot(33:64,V(33:64,j),'ko-')
  k=k+1;
  subplot(3,3,k) 
  plot(65:96,V(65:96,j),'ko-')
end

% subplot(3,3,1), set(gca,'Xlim',[1 12],'Fontsize',[14]), title('ACDC (Hard Rock)') 
% subplot(3,3,2), set(gca,'Xlim',[13 24],'Fontsize',[14]), title('Third Day (CCM)')
% subplot(3,3,3), set(gca,'Xlim',[25 36],'Fontsize',[14]), title('John Williams (Classical)') 
% subplot(3,3,4), set(gca,'Xlim',[1 12],'Fontsize',[14]) 
% subplot(3,3,5), set(gca,'Xlim',[13 24],'Fontsize',[14])
% subplot(3,3,6), set(gca,'Xlim',[25 36],'Fontsize',[14])
% subplot(3,3,7), set(gca,'Xlim',[1 12],'Fontsize',[14]) 
% subplot(3,3,8), set(gca,'Xlim',[13 24],'Fontsize',[14])
% subplot(3,3,9), set(gca,'Xlim',[25 36],'Fontsize',[14])

%% 3D version of V analyzation 

figure(5)
plot3(V(1:32,1),V(1:32,2),V(1:32,3),'ko','Linewidth',[2]) 
hold on
plot3(V(33:64,1),V(33:64,2),V(33:64,3),'ro','Linewidth',[2])
hold on
plot3(V(65:96,1),V(65:96,2),V(65:96,3),'go','Linewidth',[2])
% hold on
% plot3(V(85:112,1),V(85:112,2),V(85:112,3),'bo','Linewidth',[2])
grid on, xlabel('V1'), ylabel('V2'), zlabel('V3')

%% Classification Task

band1prelda = []; band2prelda = []; band3prelda = []; band4prelda = [];
band1prenb = []; band2prenb = []; band3prenb = []; band4prenb = [];
band1pretree = []; band2pretree = []; band3pretree = []; band4pretree = [];
n=1000

for k=1:n
    %pick a new randomization for each test.
q1 = randperm(32); q2 = randperm(32); q3 = randperm(32); q4 = randperm(32);

xband1 = V(1:32,1:15); xband2 = V(33:64,1:15);
xband3 = V(65:96,1:15); 
% xband4 = V(85:112,1:10);

xtrain = [xband1(q1(1:25),:); xband2(q2(1:25),:); xband3(q3(1:25),:)];
% xtrain2 = [xband1(q1(1:21),:); xband3(q2(1:21),:); xband4(q4(1:21),:)];
xtest = [xband1(q1(26:end),:); xband2(q2(26:end),:); xband3(q3(26:end),:)];
% xtest2 = [xband1(q1(22:end),:); xband3(q3(22:end),:); xband4(q4(22:end),:)];
ctrain = [ones(25,1); 2*ones(25,1); 3*ones(25,1)];
% ctrain2 = [ones(21,1); 2*ones(21,1); 3*ones(21,1)];

%the classifiers (LDA,Naive Bayes, CART)
[predictlda] = classify(xtest,xtrain,ctrain);
nb = fitcnb(xtrain,ctrain);
predictnb = nb.predict(xtest);
tree=fitctree(xtrain,ctrain);
predicttree = predict(tree,xtest);


% Collect the results
band1prelda = [band1prelda predictlda(1:7)];
band2prelda = [band2prelda predictlda(8:14)];
band3prelda = [band3prelda predictlda(15:21)];
% band4prelda = [band4prelda predictlda(22:28)];

band1prenb = [band1prenb predictnb(1:7)];
band2prenb = [band2prenb predictnb(8:14)];
band3prenb = [band3prenb predictnb(15:21)];
% band4prenb = [band4prenb predictnb(22:28)];

band1pretree = [band1pretree predicttree(1:7)];
band2pretree = [band2pretree predicttree(8:14)];
band3pretree = [band3pretree predicttree(15:21)];
% band4pretree = [band4pretree predicttree(22:28)];

end

success1lda = sum(band1prelda(:) == 1)/(n*7);
success2lda = sum(band2prelda(:) == 2)/(n*7);
success3lda = sum(band3prelda(:) == 3)/(n*7);
% success4lda = sum(band4prelda(:) == 4)/(n*7);

success1nb = sum(band1prenb(:) == 1)/(n*7);
success2nb = sum(band2prenb(:) == 2)/(n*7);
success3nb = sum(band3prenb(:) == 3)/(n*7);
% success4nb = sum(band4prenb(:) == 4)/(n*7);

success1tree = sum(band1pretree(:) == 1)/(n*7);
success2tree = sum(band2pretree(:) == 2)/(n*7);
success3tree = sum(band3pretree(:) == 3)/(n*7);
% success4tree = sum(band4pretree(:) == 4)/(n*7);


%% Display classification Results

ldaresults = [success1lda success2lda success3lda]
nbresults = [success1nb success2nb success3nb]
treeresults = [success1tree success2tree success3tree]

figure(6)
subplot(1,3,1)
histogram(band1prelda,3,'BinWidth',.8)
axis([.8 3 0 7000]), xticks([1.2 2 2.7]),xticklabels({'AK47', 'FAL', 'M4'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]), 
title('AK-47 Samples'), text(2,6800,'Accuracy='+string(success1lda*100)+'%','Fontsize',[12])
subplot(1,3,2)
histogram(band2prelda,3,'BinWidth',.8), axis([1 3 0 7000])
xticks([1.2 2 2.7]),xticklabels({'AK47', 'FAL', 'M4'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]), 
title('FAL Samples'), text(2,6800,'Accuracy='+string(success2lda*100)+'%','Fontsize',[12])
subplot(1,3,3)
histogram(band3prelda,3,'BinWidth',.8), axis([1 3 0 7000])
xticks([1.2 2 2.7]),xticklabels({'AK47', 'FAL', 'M4'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]),
title('M4 Samples'), text(2,6800,'Accuracy='+string(success3lda*100)+'%','Fontsize',[12])
% subplot(1,4,4)
% histogram(band4pretree,4), axis([1 4 0 7000])
% xticks([1.2 2 3 3.8]),xticklabels({'Bluegrass', 'CCM', 'Classical', 'Army Songs'})
% set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12])







       