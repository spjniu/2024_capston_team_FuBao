/*
1. 한쪽 발에서의 전족의 비율
2. 한쪽 발에서의 중+후족의 비율
3. 양발(왼발과 오른발)의 압력 비율 비교
05.12 무게 비율 구하기까지 성공 위에 주석 처리돼있는 것들 처리 필요
05.14 캘리브레이션 결과 lb= -7000 lf= -9400 rf= -10200 rb= -10500  
캘리브레이션 진행 됐으나 판자를 조금씩만 움직여도 수치가 변하는 현상 발생 일단 이 값으로 진행 후 마지막에 외관 본드로 완전히 붙히고 캘리브레이션 재진행 필요
05.25 UI에서 continue 누르면 다시 압력 측정 continue 누르면 압력 측정 다시 진행필요.
-> 라즈베리파이에서 continue 누르면 continue 오도록, quit 누르면 quit 오도록 수정 필요
*/

#include "HX711.h"


//05.23 캘리브레이션 결과 lb= -19500 lf= -20500 rf= -20000 rb= -19500 
//06.13 외관 가운데 메꾼 후 최종 캘리브레이션
// HX711 circuit wiring 각 HX711들의 포트 지정 back은 후족, front는 중족,전족 의미
const int LOADCELL_left_back_DOUT_PIN = 2;
const int LOADCELL_left_back_SCK_PIN = 3;
const int LOADCELL_left_front_DOUT_PIN = 4;
const int LOADCELL_left_front_SCK_PIN = 5;
const int LOADCELL_right_back_DOUT_PIN = 6;
const int LOADCELL_right_back_SCK_PIN = 7;
const int LOADCELL_right_front_DOUT_PIN = 8;
const int LOADCELL_right_front_SCK_PIN = 9;

long flag;
long continue_flag; // continue 신호 받으면 continue_flag= 1로 변환 기본값은 0
long loop_flag;
long start_flag;
long back_flag;
long back_during_flag;
String exercise_name="";
String exercise="";
float sum_weight=0;
long upload_flag=0;
int weight=0;

HX711 scale_left_back; 
HX711 scale_left_front;
HX711 scale_right_back; 
HX711 scale_right_front;

// 각각의 수치들을 관리하는 구조체
struct mean
{
  long lb=0; // 왼쪽 후족 측정한 수치들의 평균
  long lf=0; // 왼쪽 전족 측정한 수치들의 평균
  long rb=0; // 오른쪽 후족 측정한 수치들의 평균
  long rf=0; // 오른쪽 전족 측정한 수치들의 평균
  long rate_left_front=0; // 왼쪽 전체 중에서 전족의 비율
  long rate_left_back=0; // 왼쪽 전체 중에서 후족의 비율
  long rate_right_front=0; // 오른쪽 전체 중에서 전족의 비율
  long rate_right_back=0; // 오른쪽 전체 중에서 후족의 비율
  long rate_both_left=0; // 양쪽발 전체에 실린 압력 중 왼쪽 발의 비율
  long rate_both_right=0; // 양쪽발 전체에 실린 압력중 오른쪽 발의 비율 
  long sum_weight=0;
};


void setup() 
{ 
  Serial.begin(19200);


  scale_left_back.begin(LOADCELL_left_back_DOUT_PIN, LOADCELL_left_back_SCK_PIN);
  scale_left_front.begin(LOADCELL_left_front_DOUT_PIN, LOADCELL_left_front_SCK_PIN);
  scale_right_back.begin(LOADCELL_right_back_DOUT_PIN, LOADCELL_right_back_SCK_PIN);
  scale_right_front.begin(LOADCELL_right_front_DOUT_PIN, LOADCELL_right_front_SCK_PIN);

//05.23 캘리브레이션 결과 lb= -19500 lf= -20500 rf= -20000 rb= -19500 
  scale_left_back.set_scale(-20000 ); // 각자 calibration 해보고 구한 값 넣기
  scale_left_back.tare();  // 현재 값을 0으로 정한다는 코드
 
  scale_left_front.set_scale(-20000); // 각자 calibration 해보고 구한 값 넣기
  scale_left_front.tare();  // 현재 값을 0으로 정한다는 코드


  scale_right_back.set_scale(-19800); // 각자 calibration 해보고 구한 값 넣기
  scale_right_back.tare();  // 현재 값을 0으로 정한다는 코드


  scale_right_front.set_scale( -20100); // 각자 calibration 해보고 구한 값 넣기
  scale_right_front.tare();  // 현재 값을 0으로 정한다는 코드
  //flag=0;
  continue_flag=0;
  start_flag=0;
  Serial.println("Zero adjustment completed");

}


void loop() 
{
  int n = 0 ; // n은 운동 하는 시간동안 측정된 수치들의 갯수
  int serial_flag=0; // 시리얼 통신을 통한 값을 받아야 넘어가는 블럭 역할
  long sum_lb =0 ; // 왼쪽 후족 압력 전체의 합
  long sum_lf =0 ; // 왼쪽 전족 압력 전체의 합
  long sum_rb =0 ; // 오른쪽 후족 압력 전체의 합
  long sum_rf =0 ; // 오른쪽 전족 압력 전체의 합
  struct mean means; // 함수에서 사용할 구조체 선언
  sum_weight=0;
  if(back_during_flag==1) // 운동중 페이지에서 back 신호 받은 것 처리
  {
    serial_flag=1;
  }
  if(back_flag==1) // 무게입력 페이지에서 back 신호 받은 것 처리
  {
    serial_flag=0;
  }

  // 운동 선택 페이지
  while(serial_flag==0){
    if(continue_flag==0){
      if (Serial.available() > 0) {
          String exercise = Serial.readStringUntil('\n');  // 라즈베리 파이로부터 데이터 읽기
          Serial.print("Exercise: "); 
          Serial.println(exercise); 
          exercise_name=exercise;
          serial_flag=1;
          start_flag=1;
      }
    }
    else if(continue_flag==1){
      serial_flag++;
    }
  }
  // 무게 입력 받는 페이지
  while(serial_flag==1){
    if(continue_flag==0){
      if (Serial.available() > 0) {
        String received_data = Serial.readStringUntil('\n');  // 라즈베리 파이로부터 데이터 읽기
        if(received_data == "back")
        {
          serial_flag-=1;
          start_flag-=1;
          back_flag=1;
          Serial.print("Serial_flag:");
          Serial.println(serial_flag);
          Serial.print("Continue_flag:");
          Serial.println(continue_flag);
        }
        else
        {
          weight = received_data.toInt();  // weight 값을 읽어와 변수에 저장
          if(weight>0){ 
            Serial.print("Weight: "); 
            Serial.println(weight);
            serial_flag=2;
            start_flag=1;
          }
        }
      }
    }
    else if(continue_flag==1){
      serial_flag++;
      Serial.println(weight);
    }
  }
  // 측정 시작 -> 몸무게 받고 시작 누르면 START
  while(serial_flag==2)
  {
    if(continue_flag==0)
    {
      if (Serial.available() > 0) {
        String received_data = Serial.readStringUntil('\n');
        if (received_data == "start") {
          Serial.println("Start now"); 
          delay(1000);
          serial_flag=3;
          back_during_flag=0; // back_flag off 시키기
          back_flag=0;
        }
      }
    }
    else if(continue_flag==1)
    {
      Serial.println("Start now"); 
      Serial.println(weight);
      delay(1000);
      serial_flag=3;

    }
  }

  while(serial_flag==3) // 압력 측정 끝나면 ui 에서 결과 받기 누르면 end 받기
  { 
    float reading_left_back_round = scale_left_back.get_units() * 0.453592;
    String reading_left_back_str = String(reading_left_back_round, 1); 
    float reading_left_back = reading_left_back_str.toFloat();
    float reading_left_front_round = scale_left_front.get_units() * 0.453592;
    String reading_left_front_str = String(reading_left_front_round, 1); 
    float reading_left_front = reading_left_front_str.toFloat();

    float reading_right_back_round = scale_right_back.get_units() * 0.453592;
    String reading_right_back_str = String(reading_right_back_round, 1); 
    float reading_right_back = reading_left_back_str.toFloat();

    float reading_right_front_round = scale_right_front.get_units() * 0.453592;
    String reading_right_front_str = String(reading_right_front_round, 1); 
    float reading_right_front = reading_right_front_str.toFloat();
    

    sum_weight=reading_left_back+reading_left_front+reading_right_back+reading_right_front;
    if((reading_left_back >0.0) && (reading_left_front >0.0) && (reading_right_back >0.0) && (reading_right_front >0.0) && (sum_weight<weight) )
    {
      sum_lb = reading_left_back+sum_lb ;
      Serial.print("HX711 left back reading: ");
      Serial.print(reading_left_back,1);
      Serial.println(" kg ");

      sum_lf = reading_left_front+sum_lf;
      Serial.print("HX711 left front reading: ");
      Serial.print(reading_left_front,1);
      Serial.println(" kg ");

      sum_rb = reading_right_back+sum_rb ;
      Serial.print("HX711 right back reading: ");
      Serial.print(reading_right_back,1);
      Serial.println(" kg ");

      sum_rf = reading_right_front+sum_rf;
      Serial.print("HX711 right front reading: ");
      Serial.print(reading_right_front,1);
      Serial.println(" kg ");
      n++;
    }
  if (Serial.available() > 0) {
    String received_data = Serial.readStringUntil('\n'); 
    if(received_data == "back")
    {
      delay(500);  
      serial_flag=1;
      start_flag-=1; // start_flag=0
      back_during_flag=1;
      Serial.println(start_flag);
      Serial.println("back is done");
    }
  else if (received_data == "end")
  {
    Serial.println("End");
    delay(200); 
    serial_flag=4;

    means.sum_weight=sum_lb+sum_rb+sum_lf+sum_rf;
    means.lb=sum_lb/n;// lb= 왼발 후면부
    means.lf=sum_lf/n;// lf= 왼발 전면부
    means.rb=sum_rb/n;// rb= 오른발 후면부
    means.rf=sum_rf/n;// rf= 오른발 전면부

    if(means.sum_weight==0)
    {
    Serial.println("Error: There is no enough weight");
    }

        // 1. 왼발 앞뒤 상대비율
        means.rate_left_front=means.lf*100/(means.lf+means.lb);
        means.rate_left_back=100-means.rate_left_front;
        //2. 오른발 앞뒤 상대비율
        means.rate_right_front=means.rf*100/(means.rf+means.rb);
        means.rate_right_back=100-means.rate_right_front;
        //3. 좌우 상대비율
        means.rate_both_left=(means.lf+means.lb)*100/(means.lf+means.lb+means.rb+means.rf);
        means.rate_both_right=100-means.rate_both_left;


        Serial.print("Exercise: "); 
        Serial.println(exercise_name);

          delay(1000);
          Serial.print("Weight_sum: ");
          Serial.println(means.sum_weight);
          Serial.print("rate_left_front: ");
          Serial.println(means.rate_left_front);
          Serial.print("rate_left_back: ");
          Serial.println(means.rate_left_back);
          Serial.print("rate_right_front: ");
          Serial.println(means.rate_right_front);
          Serial.print("rate_right_back: ");
          Serial.println(means.rate_right_back);
          Serial.print("rate_both_left: ");
          Serial.println(means.rate_both_left);
          Serial.print("rate_both_right: ");
          Serial.println(means.rate_both_right);

    }
  }
  }
  // here is result page until press continue,quit,back
    while((((continue_flag%2)==1)||(start_flag==1))&&(upload_flag==0)){
    if (Serial.available() > 0) {
      String received_data = Serial.readStringUntil('\n');
      Serial.println("selected continue or quit or back");
      if (received_data == "continue") {
        delay(500); 
        continue_flag=1;
        break;

      } 
      else if (received_data == "quit") {
        delay(500); 

        start_flag = 0;
        continue_flag=0;
      }
      else if (received_data == "back")
      {
        continue_flag=0;
        serial_flag=0;
        Serial.print("Serial_flag:");
        Serial.println(serial_flag);
        Serial.print("Start_flag:");
        Serial.println(start_flag);
        Serial.print("Continue_flag:");
        Serial.println(continue_flag);
        continue_flag=0;
        break; 
      }

      }
  }
}




 
