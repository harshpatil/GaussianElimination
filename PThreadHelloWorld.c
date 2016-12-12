#include <printf.h>
#include <pthread.h>
//
// Created by Harshavardhan Patil on 9/30/16.
//
int i=1;
void* sayHello(void *foo){
    printf("%d :: Hello world, this is my first Pthread program\n", i++);
    return NULL;
}
/*

int main(){

    pthread_t pthread[16];
    int i;
    for(i=0; i<16; i++){
        pthread_create(&pthread[i], NULL, sayHello, NULL);
        printf("Thread id :: %d \n", pthread[i]);
    }
    for(i=0; i<16; i++){
        pthread_join(&pthread[i], NULL);
    }
    printf("Finished\n");
    return 0;
}*/
