%% Compiles elements of test data into individual matrices
clear all; close all; clc
     mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Combined Files\AK47\1k spectrograms';

    D=dir(fullfile(mainpath,'*cropped.jpg'));  
Band1Samp=[];
Band1AveSamp=[];

    for j=1:numel(D)
           song = imread(fullfile(mainpath,D(j).name));
         blurspectro = double(imresize(song, [64,64]));
%          imshow(blurspectro)
         avespectro = double(blurspectro) - mean(double(blurspectro));
         skinnyspectro =reshape(blurspectro,1,4096);
         skinnyavespectro = reshape(avespectro,1,4096);
         Band1AveSamp = [skinnyavespectro' Band1AveSamp];
         Band1Samp = [skinnyspectro' Band1Samp];
    end
    
     mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Combined Files\FAL\1k spectrograms';
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
         mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Combined Files\M4\1k spectrograms';
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
         mainpath = 'C:\Users\gavin\Documents\MATLAB\ProjectData\Combined Files\50 cal\1k spectrograms';
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
    
    band1_wave = dc_wavelet(Band1AveSamp);
    band2_wave = dc_wavelet(Band2AveSamp);
    band3_wave = dc_wavelet(Band3AveSamp);
    band4_wave = dc_wavelet(Band4AveSamp);
    
    [U,S,V]=svd([band1_wave,band2_wave,band3_wave,band4_wave],0);
    %% U Analysis
    
    figure(2)
for j=1:12
  subplot(3,4,j) 
  ut1 = reshape(U(:,j),32,32); 
  ut2 = ut1(32:-1:1,:); 
  pcolor(ut2), colormap(jet)
  set(gca,'Xtick',[],'Ytick',[])
  colorbar
    title('Mode ' + string(j))
end

%% Analyze S

figure(3)
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14]) 
axis([0 630 0 4.6*10^4])
yticks([5000 10000 15000 20000 25000 30000 35000 40000 45000])
xlabel('Singular Values')
ylabel('\sigma')
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14])
axis([0 630 0 4.8*10^4])
xlabel('Singular Values')
ylabel('\sigma')

%Singular value assessment
sig=diag(S);
energy1=sig(1)/sum(sig)
energy2=sum(sig(1:27))/sum(sig) 
eng10perc=sum(sig(1:63))/sum(sig)
eng25perc=sum(sig(1:158))/sum(sig)
eng50perc=sum(sig(1:315))/sum(sig)
eng=sum(sig(1:404))/sum(sig)

%% 3D version of V analyzation 

figure(5)
plot3(V(1:190,1),V(1:190,2),V(1:190,3),'ko','Linewidth',[2]) 
hold on
plot3(V(191:294,1),V(191:294,2),V(191:294,3),'ro','Linewidth',[2])
hold on
plot3(V(295:465,1),V(295:465,2),V(295:465,3),'go','Linewidth',[2])
hold on
plot3(V(466:629,1),V(466:629,2),V(466:629,3),'bo','Linewidth',[2])
grid on, xlabel('V1'), ylabel('V2'), zlabel('V3')
legend('AK-47','FN FAL', 'M4', '50 cal')

%% Classification Task

band1prelda = []; band2prelda = []; band3prelda = []; band4prelda = [];
band1prenb = []; band2prenb = []; band3prenb = []; band4prenb = [];
band1pretree = []; band2pretree = []; band3pretree = []; band4pretree = [];
n=100

lda1 = []; lda2 = []; lda3 = []; lda4 = []; 
nb1 = []; nb2 = []; nb3 = []; nb4 = []; 
tr1 = []; tr2 = []; tr3 = []; tr4 = []; 
for p=1:5
    band1prelda = []; band2prelda = []; band3prelda = []; band4prelda = [];
band1prenb = []; band2prenb = []; band3prenb = []; band4prenb = [];
band1pretree = []; band2pretree = []; band3pretree = []; band4pretree = [];
for k=1:n
    %pick a new randomization for each test.
q1 = randperm(190); q2 = randperm(104); q3 = randperm(171); q4 = randperm(164);

xband1 = V(1:190,1:27); xband2 = V(191:294,1:27);
xband3 = V(295:465,1:27); xband4 = V(466:629,1:27);

xtrain = [xband1(q1(1:183),:); xband2(q2(1:97),:); xband3(q3(1:164),:); xband4(q4(1:157),:)];
xtest = [xband1(q1(184:end),:); xband2(q2(98:end),:); xband3(q3(165:end),:); xband4(q4(158:end),:)];
ctrain = [ones(183,1); 2*ones(97,1); 3*ones(164,1); 4*ones(157,1)];

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
band4prelda = [band4prelda predictlda(22:28)];

band1prenb = [band1prenb predictnb(1:7)];
band2prenb = [band2prenb predictnb(8:14)];
band3prenb = [band3prenb predictnb(15:21)];
 band4prenb = [band4prenb predictnb(22:28)];

band1pretree = [band1pretree predicttree(1:7)];
band2pretree = [band2pretree predicttree(8:14)];
band3pretree = [band3pretree predicttree(15:21)];
 band4pretree = [band4pretree predicttree(22:28)];

end

success1lda = sum(band1prelda(:) == 1)/(n*7);
success2lda = sum(band2prelda(:) == 2)/(n*7);
success3lda = sum(band3prelda(:) == 3)/(n*7);
success4lda = sum(band4prelda(:) == 4)/(n*7);

success1nb = sum(band1prenb(:) == 1)/(n*7);
success2nb = sum(band2prenb(:) == 2)/(n*7);
success3nb = sum(band3prenb(:) == 3)/(n*7);
success4nb = sum(band4prenb(:) == 4)/(n*7);

success1tree = sum(band1pretree(:) == 1)/(n*7);
success2tree = sum(band2pretree(:) == 2)/(n*7);
success3tree = sum(band3pretree(:) == 3)/(n*7);
success4tree = sum(band4pretree(:) == 4)/(n*7);

lda1 = [lda1 success1lda]; lda2 = [lda2 success2lda]; lda3 = [lda3 success3lda]; lda4 = [lda4 success4lda]; 
nb1 = [nb1 success1nb]; nb2 = [nb2 success2nb]; nb3 = [nb3 success3nb]; nb4 = [nb4 success4nb]; 
tr1 = [tr1 success1tree]; tr2 = [tr2 success2tree]; tr3 = [tr3 success3tree]; tr4 = [tr4 success4tree]; 

end

lda1
lda2
lda3
lda4
nb1
nb2
nb3
nb4
tr1
tr2
tr3
tr4

%% Display classification Results

ldaresults = [success1lda success2lda success3lda success4lda]
nbresults = [success1nb success2nb success3nb success4nb]
treeresults = [success1tree success2tree success3tree success4tree]

figure(6)
subplot(1,4,1)
histogram(band1prelda,4,'BinWidth',.8)
axis([.8 4 0 7000]), xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]), 
title('AK-47 Samples'), text(1.7,6800,'Accuracy='+string(round(success1lda*100,1))+'%','Fontsize',[12])
subplot(1,4,2)
histogram(band2prelda,4,'BinWidth',.8), axis([1 4 0 7000])
xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]), 
title('FAL Samples'), text(1.7,6800,'Accuracy='+string(round(success2lda*100,1))+'%','Fontsize',[12])
subplot(1,4,3)
histogram(band3prelda,4,'BinWidth',.8), axis([1 4 0 7000])
xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]),
title('M4 Samples'), text(1.7,6800,'Accuracy='+string(round(success3lda*100,1))+'%','Fontsize',[12])
subplot(1,4,4)
histogram(band4prelda,4,'BinWidth',.8), axis([1 4 0 7000])
xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12])
title('M2 50 cal Samples'), text(1.7,6800,'Accuracy='+string(round(success4lda*100,1))+'%','Fontsize',[12])


%% low V CART Results
 band1prelda = []; band2prelda = []; band3prelda = []; band4prelda = [];
band1prenb = []; band2prenb = []; band3prenb = []; band4prenb = [];
band1pretree = []; band2pretree = []; band3pretree = []; band4pretree = [];
n=1000
for k=1:n
    %pick a new randomization for each test.
q1 = randperm(190); q2 = randperm(104); q3 = randperm(171); q4 = randperm(164);

xband1 = V(1:190,1:4); xband2 = V(191:294,1:4);
xband3 = V(295:465,1:4); xband4 = V(466:629,1:4);

xtrain = [xband1(q1(1:183),:); xband2(q2(1:97),:); xband3(q3(1:164),:); xband4(q4(1:157),:)];
xtest = [xband1(q1(184:end),:); xband2(q2(98:end),:); xband3(q3(165:end),:); xband4(q4(158:end),:)];
ctrain = [ones(183,1); 2*ones(97,1); 3*ones(164,1); 4*ones(157,1)];

%the classifiers (LDA,Naive Bayes, CART)
tree=fitctree(xtrain,ctrain);
predicttree = predict(tree,xtest);

% Collect the results
band1pretree = [band1pretree predicttree(1:7)];
band2pretree = [band2pretree predicttree(8:14)];
band3pretree = [band3pretree predicttree(15:21)];
 band4pretree = [band4pretree predicttree(22:28)];
end

success1tree = sum(band1pretree(:) == 1)/(n*7);
success2tree = sum(band2pretree(:) == 2)/(n*7);
success3tree = sum(band3pretree(:) == 3)/(n*7);
success4tree = sum(band4pretree(:) == 4)/(n*7);

figure(7)
subplot(1,4,1)
histogram(band1pretree,4,'BinWidth',.8)
axis([.8 4 0 7000]), xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]), 
title('AK-47 Samples'), text(1.7,6800,'Accuracy='+string(round(success1tree*100,1))+'%','Fontsize',[12])
subplot(1,4,2)
histogram(band2pretree,4,'BinWidth',.8), axis([1 4 0 7000])
xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]), 
title('FAL Samples'), text(1.7,6800,'Accuracy='+string(round(success2tree*100,1))+'%','Fontsize',[12])
subplot(1,4,3)
histogram(band3pretree,4,'BinWidth',.8), axis([1 4 0 7000])
xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12]),
title('M4 Samples'), text(1.7,6800,'Accuracy='+string(round(success3tree*100,1))+'%','Fontsize',[12])
subplot(1,4,4)
histogram(band4pretree,4,'BinWidth',.8), axis([1 4 0 7000])
xticks([1.2 2 3 3.8]),xticklabels({'AK47', 'FAL', 'M4','50cal'})
set(gca,'YTick',[1000 2000 3000 4000 5000 6000 7000], 'Fontsize',[12])
title('M2 50 cal Samples'), text(1.7,6800,'Accuracy='+string(round(success4tree*100,1))+'%','Fontsize',[12])
view(tree,'Mode','graph')


     