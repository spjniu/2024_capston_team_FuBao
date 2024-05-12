/*
 UI 개발할 때 set 페이지에서 start 누르면 압력감지가 시작되도록 연동
1. 한쪽 발에서의 전족의 비율
2. 한쪽 발에서의 중+후족의 비율
3. 양발(왼발과 오른발)의 압력 비율 비교
카메라랑 연동해서 횟수 끝나면 운동 종료 or 무게 관련 알고리즘으로 바벨 들면 시작 내려놓으면 종료 그때까지 모든 시점 값 다 더해서 평균내기
원래 무게에서 바벨 들어서 무게가 증가되면 시작, 원래 무게로 돌아오면 종료
*/
// 05.12 무게 비율 구하기까지 성공 위에 주석 처리돼있는 것들만 완성하면 될듯
#include "HX711.h"
 
//lb= -7000 lf= -7000 , rb= -7000 , rf= -7000 캘리브레이션 값 외관 완성 후 다시 재검토 필요
// HX711 circuit wiring 각 HX711들의 포트 지정 back은 후족, front는 중족,전족 의미
const int LOADCELL_left_back_DOUT_PIN = 2;
const int LOADCELL_left_back_SCK_PIN = 3;
const int LOADCELL_left_front_DOUT_PIN = 4;
const int LOADCELL_left_front_SCK_PIN = 5;
const int LOADCELL_right_back_DOUT_PIN = 6;
const int LOADCELL_right_back_SCK_PIN = 7;
const int LOADCELL_right_front_DOUT_PIN = 8;
const int LOADCELL_right_front_SCK_PIN = 9;


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
};
// 구조체를 리턴하는 함수
struct mean calculate_mean()
{
  int n = 0 ; // n은 운동 하는 시간동안 측정된 수치들의 갯수
  long sum_lb =0 ; // 왼쪽 후족 압력 전체의 합
  long sum_lf =0 ; // 왼쪽 전족 압력 전체의 합
  long sum_rb =0 ; // 오른쪽 후족 압력 전체의 합
  long sum_rf =0 ; // 오른쪽 전족 압력 전체의 합
  struct mean means; // 함수에서 사용할 구조체 선언


  while(n<=10) // 테스트 용으로 n <= 10 으로 사용. 측정 시작과 끝 확실히 해야 함.
  {
    if (scale_left_back.is_ready()) 
    {
    long reading_left_back = scale_left_back.get_units()*0.453592;
    sum_lb = reading_left_back+sum_lb ;
    
    Serial.print("HX711 left back reading: ");
    Serial.print(reading_left_back);
    Serial.println(" kg ");
    
    } 
  delay(300);
    if (scale_left_front.is_ready()) 
    {
    long reading_left_front = scale_left_front.get_units()*0.453592;
    sum_lf = reading_left_front+sum_lf;
    
    Serial.print("HX711 left front reading: ");
    Serial.print(reading_left_front);
    Serial.println(" kg ");
    
    } 
  delay(300);
    if (scale_right_back.is_ready()) 
    {
    long reading_right_back = scale_right_back.get_units()*0.453592;
    sum_rb = reading_right_back+sum_rb ;
    
    Serial.print("HX711 right back reading: ");
    Serial.print(reading_right_back);
    Serial.println(" kg ");
    
    }   
  delay(300);
    if (scale_right_front.is_ready()) 
    {
    long reading_right_front = scale_right_front.get_units()*0.453592;
    sum_rf = reading_right_front+sum_rf;
    
    Serial.print("HX711 right front reading: ");
    Serial.print(reading_right_front);
    Serial.println(" kg ");
    
    } 
  delay(300);
  n++ ; // 한번 다 측정되면 n이 1증가 평균구할 때 n으로 나누기
}
//무게를 조금 더 정확하게 측정하기 위해 여러 수치에 대하여 평균으로 측정
means.lb=sum_lb/n;
means.lf=sum_lf/n;
means.rb=sum_rb/n;
means.rf=sum_rf/n;

// 1. 왼발 앞뒤 상대비율
means.rate_left_front=means.lf*100/(means.lf+means.lb);
means.rate_left_back=100-means.rate_left_front;
//2. 오른발 앞뒤 상대비율
means.rate_right_front=means.rf*100/(means.rf+means.rb);
means.rate_right_back=100-means.rate_right_front;
//3. 좌우 상대비율
means.rate_both_left=(means.lf+means.lb)*100/(means.lf+means.lb+means.rb+means.rf);
means.rate_both_right=100-means.rate_both_left;

Serial.print(" 함수 내 왼발 후족의 비율: ");
Serial.print(means.rate_left_back);
Serial.println("%");
Serial.print(" 함수 내 왼발 전족의 비율: ");
Serial.print(means.rate_left_front);
Serial.println("%");
Serial.print("함수 내 오른발 후족의 비율: ");
Serial.print(means.rate_right_back);
Serial.println("%");
Serial.print("함수 내 오른발 전족의 비율: ");
Serial.print(means.rate_right_front);
Serial.println("%");
Serial.print("함수 내 전체 중 왼발의 비율: ");
Serial.print(means.rate_both_left);
Serial.println("%");
Serial.print("함수 내 전체 중 오른발의 비율: ");
Serial.print(means.rate_both_right);
Serial.println("%");

return means; // sum_lb mean_lb로 바꾸기 나중에 회로 고쳐지면 ( ?? )
}

void setup() 
{
  Serial.begin(9600);
  scale_left_back.begin(LOADCELL_left_back_DOUT_PIN, LOADCELL_left_back_SCK_PIN);
  scale_left_front.begin(LOADCELL_left_front_DOUT_PIN, LOADCELL_left_front_SCK_PIN);
  scale_right_back.begin(LOADCELL_right_back_DOUT_PIN, LOADCELL_right_back_SCK_PIN);
  scale_right_front.begin(LOADCELL_right_front_DOUT_PIN, LOADCELL_right_front_SCK_PIN);


  scale_left_back.set_scale(-10000 ); // 각자 calibration 해보고 구한 값 넣기
  scale_left_back.tare();  // 현재 값을 0으로 정한다는 코드
 
  scale_left_front.set_scale(-10000); // 각자 calibration 해보고 구한 값 넣기
  scale_left_front.tare();  // 현재 값을 0으로 정한다는 코드


  scale_right_back.set_scale(-10000); // 각자 calibration 해보고 구한 값 넣기
  scale_right_back.tare();  // 현재 값을 0으로 정한다는 코드


  scale_right_front.set_scale( -10000); // 각자 calibration 해보고 구한 값 넣기
  scale_right_front.tare();  // 현재 값을 0으로 정한다는 코드

  Serial.println("Start Now");
  delay(2000);
}

/*
 UI 개발할 때 set 페이지에서 start 누르면 압력감지가 시작되도록 연동
1. 한쪽 발에서의 전족의 비율
2. 한쪽 발에서의 중+후족의 비율
3. 양발(왼발과 오른발)의 압력 비율 비교
카메라랑 연동해서 횟수 끝나면 운동 종료 or 무게 관련 알고리즘으로 바벨 들면 시작 내려놓으면 종료 그때까지 모든 시점 값 다 더해서 평균내기
원래 무게에서 바벨 들어서 무게가 증가되면 시작, 원래 무게로 돌아오면 종료
*/
// 05.12 무게 비율 구하기까지 성공 위에 주석 처리돼있는 것들만 완성하면 될듯



void loop() 
{

struct mean means;
means=calculate_mean(); // 리턴되는 구조체 그대로 받아오기


Serial.print("왼발 후족의 비율: ");
Serial.print(means.rate_left_back);
Serial.println("%");
Serial.print("왼발 전족의 비율: ");
Serial.print(means.rate_left_front);
Serial.println("%");
Serial.print("오른발 후족의 비율: ");
Serial.print(means.rate_right_back);
Serial.println("%");
Serial.print("오른발 전족의 비율: ");
Serial.print(means.rate_right_front);
Serial.println("%");
Serial.print("전체 중 왼발의 비율: ");
Serial.print(means.rate_both_left);
Serial.println("%");
Serial.print("전체 중 오른발의 비율: ");
Serial.print(means.rate_both_right);
Serial.println("%");


}

