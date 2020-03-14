clear all; close all; clc;

%  ***get correct folder***
files = dir('Gavin_WeaponSounds/50 cal v2/*.wav') ; 

N = length(files);   
for i = 1:N
    remove = [];
    consecutiveWhitespace = [];
    thisfile = files(i).name ; 
    [data,fs] = audioread(thisfile);
    [rows, columns] = size(data);
    if (columns == 1)
        continue;
    end
%     sound(data, fs);
    datalength = length(data);
    signalCounter = 0;
    inSignal = false;
    dataForNewFile = [];
    jValuesToRemove = [];
    for j = 1:datalength
        if (j == datalength)
            break;
        end
        testVariable = data(j,1);
        % at the end of the signal, switching to whitespace, or multiple zeros
        % inside the signal
        if (abs(data(j,1)) < 0.002 && abs(data(j,2)) < 0.002 && inSignal == true)
            consecutiveWhitespace = [consecutiveWhitespace data(j,1)];
            cwsize = size(consecutiveWhitespace,2);
            if (cwsize > 48000)%1 seconds
                %split off previous data into a new file
                newFileName = erase(thisfile,".wav");
                newFileNameChar = append(newFileName,"_", int2str(signalCounter), ".wav");
                audiowrite(newFileNameChar, dataForNewFile, fs);
                dataForNewFile = [];
                inSignal = false;
%             signalCounter = 0;
            % 1-2 zeros in signal
            else
               saveDataPoint = data(j,:);%either one or two columns
               dataForNewFile = [dataForNewFile; saveDataPoint];
               inSignal = true;
               continue; 
            end
        % in whitespace
        elseif (abs(data(j,1)) < 0.002 && inSignal == false)
            consecutiveWhitespace = [consecutiveWhitespace data(j,1)];
            inSignal = false;
        % start of a new signal
        elseif ((abs(data(j,1)) > 0.002 || abs(data(j,2)) > 0.002) && inSignal == false)
            signalCounter = signalCounter + 1;            
            consecutiveWhitespace = [];
            saveDataPoint = data(j,:);%either one or two columns
            dataForNewFile = [dataForNewFile; saveDataPoint];
            inSignal = true;
            continue;
        %inside the signal
        elseif ((abs(data(j,1)) > 0.002 || abs(data(j,2)) > 0.002) && inSignal == true)
            consecutiveWhitespace = [];
            saveDataPoint = data(j,:);%either one or two columns
            dataForNewFile = [dataForNewFile; saveDataPoint];
            inSignal = true;
            continue;
        end
    end
end
