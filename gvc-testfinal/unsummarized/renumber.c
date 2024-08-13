#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

char a[2000][100],b[2000][100],c[2000][100],d[2000][100];
double s[2000],ss[2000];
int i,n;
char kind[100];

double logrant=9e99, hideny=-9e99;
double lodeny=9e99, higrant=-9e99;

int main() {
  while (4==scanf(" %[^,],%[^,],%[^,],%s",a[i],b[i],c[i],d[i])) {
     if (!strncmp(b[i],"predict",7)) continue;
     sscanf(c[i],"%lg",&s[i]);
     //printf("%s %s %s %lf %s\n",a[i],b[i],c[i],s[i],d[i]);
     while (*b[i] == ' ') strcpy (b[i],b[i]+1);
     if (!strcmp(b[i],"grant")) {
        if (s[i] < logrant)logrant=s[i];
        if (s[i] > higrant) higrant=s[i];
     } else if (!strcmp(b[i],"deny")) {
        if (s[i] > hideny) hideny=s[i];
        if (s[i] < lodeny) lodeny=s[i];
     } else {
         printf("oops! >>%s<<\n",b[i]);
         exit(1);
     }
     i++;
  }
  strcpy(kind,"KindUnknown");
  if (lodeny < 0 && higrant > 0 && hideny < logrant) strcpy(kind,"KindLogOdds");
  if (lodeny > 0 && higrant < 1 && hideny < logrant) strcpy(kind,"KindProb");
  if (hideny >= logrant && hideny < 1 && lodeny < 1) strcpy(kind,"KindConf");
  else if (hideny >= logrant) strcpy(kind,"KindMess");
  //printf("%s logrant %lf higrant %lf lodeny %lf hideny %lf\n",kind,logrant, higrant, lodeny, hideny);
  n=i;
  for (i=0; i<n; i++) {
     ss[i] = 12345;
     if (!strcmp(kind,"KindLogOdds")) ss[i] = s[i];
     if (!strcmp(kind,"KindProb")) ss[i] = log(s[i]/(1-s[i]));
     if (!strcmp(kind,"KindConf") && !strcmp(b[i],"grant")) ss[i] = log(s[i]/(1-s[i]));
     if (!strcmp(kind,"KindConf") && !strcmp(b[i],"deny")) ss[i] = -log(s[i]/(1-s[i]));
     if (!strcmp(kind,"KindMess") && !strcmp(b[i],"grant")) ss[i] = s[i];
     if (!strcmp(kind,"KindMess") && !strcmp(b[i],"deny")) ss[i] = -s[i];
     if (ss[i] == 12345) {printf("oopsy!\n"); exit(1);}
  }
  printf("%s\n","brief,predict,score,truth");
  for (i=0;i<n;i++) {
    printf("%s,%s,%lf,%s\n",a[i],b[i],ss[i],d[i]);
  }
}
