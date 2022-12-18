clc
clear all
close all
xypos=xlsread('segment_vetex_gapped_M1CS.xlsx');
for i=411:492
    for j=1:6
    x(i,j)=xypos(i,3*j-2)*1000;
    y(i,j)=xypos(i,3*j-1)*1000;
    end
    Ax(i,:,:)=[x(i,:);y(i,:)];
    textFileName = ['F'  num2str(i-410) 'ir' '.UDA'];
       fileID = fopen(textFileName,'wt');
         fprintf(fileID,)
         fprintf(fileID,'%6.6f\t %6.6f\n',Ax(i,:));
         fprintf(fileID,'%3s\n','BRK');
         fclose(fileID); 
end