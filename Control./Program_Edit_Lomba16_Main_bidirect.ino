 #include <Servo.h>
#include <LiquidCrystal_I2C.h> // Library LiquidCrystal_I2C
#include <Wire.h> // Library I2C

#define CH1 6
#define CH3 7
#define CH5 8
#define CH6 9
#define SERVO_PIN 10 // Define servo pin is 10
#define SERVO_PIN2 11 // Define servo pin is 11
#define THRUS_PIN_A 12 // Define thruster pin is 12
#define THRUS_PIN_B 13 // Define thruster pin is 13
#define TRIGPIN 14 // Pin Trigger is 14
#define ECHOPIN 15 // Pin Echo is 15
#define TRIGPIN2 16 // Pin Trigger is 16
#define ECHOPIN2 17 // Pin Echo is 17
#define TRIGPIN3 18 // Pin Echo is 17
#define ECHOPIN3 19 // Pin Echo is 17

//Button Pin 33 nice
const int Change_buttonPin = 33; // Button change pin is 29
const int Up_buttonPin = 31; // Button up pin is 32

int ch1; // Right Horizontal
int ch3; // Left Vertical
int ch5; // Left Potensio
int ch6; // Right Potensio

int var_buttonState = 0; // Var assign digitalread button up
int var_Buttonmode = 0; // Var assign digitalread button up
int buttonPushCounter = 1488; // Thruster horizontal speed assign set value

int redPin = 3;
int greenPin = 4;
int bluePin = 5;

int Counter = 0; // Set Even and Odd to change mode
char valA = 'a'; // Initialize serial communication, initiaze as a to consider the value is char
long duration, distance, duration2, distance2, duration3, distance3; //Time to calculate distance

LiquidCrystal_I2C lcd(0x27, 16, 2); // Set I2C address and character size for LCD I2C
Servo ThrusterSpeed1;
Servo ThrusterSpeed2;
Servo ServoMove;
Servo ServoMove2;

//1. Setup
void setup() {
 //Button Value Set Point
  int buttonPushCounter = 1488; // Thruster horizontal speed assign set value
  /* LCD I2C */
  lcd.init(); // LCD initiation
  lcd.backlight(); // Turn on backlight
  lcd.begin(16, 2); // LCD size 16 x 2
  
  /*Serial Monitor*/
  Serial.begin(2000000);
  
  ServoMove.attach(SERVO_PIN, 500, 2500); // Min max servo value from calibration
  ServoMove2.attach(SERVO_PIN2, 500, 2500); // Min max servo value from calibration
  ThrusterSpeed1.attach(THRUS_PIN_A, 1000, 2000); // Min max Thruster value from calibration
  ThrusterSpeed2.attach(THRUS_PIN_B, 1000, 2000); // Min max Thruster value from calibration

  //Hardcode the servo angle setpoint */
  ServoMove.write(90);
  ServoMove2.write(90);
  ThrusterSpeed1.writeMicroseconds(buttonPushCounter); // Make sure value of thruster horizontal is 1000
  ThrusterSpeed2.writeMicroseconds(buttonPushCounter); // Make sure value of thruster horizontal is 1000
  
  //INPUT
  pinMode(CH1, INPUT);
  pinMode(CH3, INPUT);
  pinMode(CH5, INPUT);
  pinMode(CH6, INPUT);
  pinMode(Up_buttonPin , INPUT);
  pinMode(Change_buttonPin , INPUT);
  pinMode(ECHOPIN, INPUT);
  pinMode(ECHOPIN2, INPUT);
  pinMode(ECHOPIN3, INPUT);
  //OUTPUT
  pinMode(SERVO_PIN, OUTPUT); //
  pinMode(SERVO_PIN2, OUTPUT); //
  pinMode(THRUS_PIN_A, OUTPUT); //servo1
  pinMode(THRUS_PIN_B, OUTPUT); //servo
  pinMode(TRIGPIN, OUTPUT);
  pinMode(TRIGPIN2, OUTPUT);
  pinMode(TRIGPIN3, OUTPUT);
}

//2. Fungsi Ultrasonik
void UltrasSamping(){
    Serial.println("US");
    
    digitalWrite(TRIGPIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIGPIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIGPIN, LOW);
    duration = pulseIn(ECHOPIN, HIGH);
    
    digitalWrite(TRIGPIN2, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIGPIN2, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIGPIN2, LOW);
    duration2 = pulseIn(ECHOPIN2, HIGH);

    distance = duration / 58.2;
    distance2 = duration2 / 58.2;
    
    Serial.println(distance);
    Serial.println(distance2);
  
    if (distance <= 75) {
      setColor(255, 255, 0);  // Yellow
      ServoMove.write(45);
      ServoMove2.write(45);
    } 
  
    if (distance2 <= 75) {
      setColor(255, 255, 0);  // Yellow
      ServoMove.write(135);
      ServoMove2.write(135);
    }
  
    if (distance >75 && distance2 > 75){
      ServoMove.write(90);
      ServoMove2.write(90);
    }
}

void UltrasDepan(){

    Serial.println("UD");    
    digitalWrite(TRIGPIN3, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIGPIN3, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIGPIN3, LOW);
    duration3 = pulseIn(ECHOPIN3, HIGH);
    
    /*Meausre the disrance*/
    distance3 = duration3 / 58.2;
  
    if (distance3 <= 75) {
      ThrusterSpeed1.writeMicroseconds(1300); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(1300); // Make sure value of thruster horizontal is 1000
      delay(100);
      Counter=0;
      ch5=1300;
      ch6=1300;
      ServoMove.write(90);
      ServoMove2.write(90);
    }
}

//2.Fungsi RGB
void setColor(int red, int green, int blue)
{
  analogWrite(redPin, red);
  analogWrite(greenPin, green);
  analogWrite(bluePin, blue);  
}

//4. Fungsi Autonomous
void autonomous() {

  UltrasSamping();
  if (Serial.available() > 0)
  {
      UltrasSamping();
     /* Mendeteksi char */
    valA = Serial.read();
    Serial.println(valA);

    /* Range segmenting */
    if (valA == 'a')
    {
      /* Lurus lempeng */
      // Serial.println("lurus lempeng");
      Serial.println("pos servo : 90");
      ServoMove.write(90);
      ServoMove2.write(90);
      ThrusterSpeed1.writeMicroseconds(buttonPushCounter); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(buttonPushCounter); // Make sure value of thruster horizontal is 1000
    }
    if (valA == 'b')
    {
      /* Belok kiri tipis */
      // Serial.println("belok kiri tipis");
      Serial.println("pos servo : 50");
      ServoMove.write(50);
      ServoMove2.write(50);
      ThrusterSpeed1.writeMicroseconds(buttonPushCounter - 45); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(buttonPushCounter - 45); // Make sure value of thruster horizontal is 1000
    }
    if (valA == 'c')
    {
      /* Belok kanan tipis */
      // Serial.println("belok kanan tipis");
      Serial.println("pos servo : 130");
      ServoMove.write(130);
      ServoMove2.write(130);
      ThrusterSpeed1.writeMicroseconds(buttonPushCounter - 45); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(buttonPushCounter - 45); // Make sure value of thruster horizontal is 1000

    }
    if (valA == 'd')
    {
      /* Belok kiri sedang */
      // Serial.println("belok kiri sedang");
      Serial.println("pos servo : 30");
      ServoMove.write(30);
      ServoMove2.write(30);
      ThrusterSpeed1.writeMicroseconds(buttonPushCounter - 60); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(buttonPushCounter - 60); // Make sure value of thruster horizontal is 1000

    }
    if (valA == 'e')
    {
      /* Belok kanan sedang */
      // Serial.println("belok kanan sedang");
      Serial.println("pos servo : 150");
      ServoMove.write(150);
      ServoMove2.write(150);
      ThrusterSpeed1.writeMicroseconds(buttonPushCounter - 60); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(buttonPushCounter - 60); // Make sure value of thruster horizontal is 1000
    }
    if (valA == 'f')
    {
      /* Belok kiri tajam */
      // Serial.println("belok kiri tajam");
      Serial.println("pos servo : 0");
      ServoMove.write(0);
      ServoMove2.write(0);
      ThrusterSpeed1.writeMicroseconds(buttonPushCounter - 80); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(buttonPushCounter - 80); // Make sure value of thruster horizontal is 1000
    }
    if (valA == 'g')
    {
      /* Belok kanan tajam */
      // Serial.println("belok kanan tajam");
      Serial.println("pos servo : 180");
      ServoMove.write(180);
      ServoMove2.write(180);
      ThrusterSpeed1.writeMicroseconds(buttonPushCounter - 80); // Make sure value of thruster horizontal is 1000
      ThrusterSpeed2.writeMicroseconds(buttonPushCounter - 80); // Make sure value of thruster horizontal is 1000
    }
    /*Jika tidak terdeteksi bola = Kapal jalan terus*/
    if (valA == 'h')
    {
    Serial.println("Dermaga");
    ThrusterSpeed1.writeMicroseconds(1300); // Make sure value of thruster horizontal is 1000
    ThrusterSpeed2.writeMicroseconds(1300); // Make sure value of thruster horizontal is 1000
    ServoMove.write(90);
    ServoMove2.write(90);
    delay(100);
    Counter = 0;
    ch5=1300;
    ch6=1300;
    //    UltrasDepan();    
    }
  }
}
//5. Fungsi Remote
void remote()
{
  // Serial.println("mode manual");
  ch1 = pulseIn(CH1, HIGH, 30000); 
  ch3 = pulseIn(CH3, HIGH, 30000);
  ch5 = pulseIn(CH5, HIGH, 30000);
  ch6 = pulseIn(CH6, HIGH, 30000);

  //  Belok Kiri
  if (ch1 < 1200)
  { //kanan horizontal ke kiri
    ServoMove.write(30);
    ServoMove2.write(30);
  }

  //Tengah
  if (ch1 >= 1200 && ch1 <1550)
  { //kanan horizontal
    ServoMove.write(90);
    ServoMove2.write(90);
  }

  //Kanan
  if (ch1 >= 1550)
  { //kanan horizontal ke kanan
    ServoMove.write(150);
    ServoMove2.write(150);
  }

  //Maju Ke Depan
  if (ch3 > 1600 & ch3 <= 1750){ //kiri vertical ke atas
    ThrusterSpeed1.write(buttonPushCounter-50);
    ThrusterSpeed2.write(buttonPushCounter-50);
  }

  if (ch3 > 1750){ //kiri vertical ke atas
    ThrusterSpeed1.write(buttonPushCounter);
    ThrusterSpeed2.write(buttonPushCounter);
  }

/*Dikasih Gap antara berhenti dengan bergerak untuk antisipasi sinyal bocor*/
  if (ch3 >= 1300 && ch3 <= 1550){ //kiri vertical ke tengah 
    ThrusterSpeed1.write(1488);
    ThrusterSpeed2.write(1488);
  }
  
  //Berhenti
  if (ch3 < 1200){ //kiri vertical ke bawah
    ThrusterSpeed1.write(1400);
    ThrusterSpeed2.write(1400);
  }
}

//6. Fungsi Pergantian Mode Idle dengan 1 button
int checkMode(int Counter) {
  var_Buttonmode = digitalRead(Change_buttonPin); // If button is pushed
  
  //Button Normally Open (LOW=0)
  if (var_Buttonmode == HIGH) {
    if (Counter >= 0 && Counter < 10) { // Check if the counter is within the valid range
      Serial.print("number of counter pushes: "); // Write to serial Monitor
      Counter++; // Increment the counter
      Serial.println(Counter); // Write value increment ButtonPushCounter to Serial Monitor
    } 
  else if (Counter >= 10) {
      Counter = 0; // Reset the counter if it exceeds the valid range
    }
    delay(500);
  }
//  if (var_Buttonmode == LOW){
//    Counter;
//  }
  return Counter;
}

//7. Fungsi button naik
int SpeedUp(int buttonPushCounter){
  
  //Button Normally open (LOW = 0)
  var_buttonState = digitalRead(Up_buttonPin); // If button is pushed
    if (var_buttonState == LOW)
    {
      if (buttonPushCounter <= 2000)
      {
        Serial.print("number of button pushes: "); // Write to serial Monitor
        buttonPushCounter = buttonPushCounter += 100; //Write value increment ButtonPushCounter
        Serial.println(buttonPushCounter); //Write value increment ButtonPushCounter to Serial Monitor
        delay(500);
      }
      if (buttonPushCounter > 2000) 
      {
        buttonPushCounter = 1488;
      }
    }
   return buttonPushCounter;
  }

//8. Fungsi Idle Speed Menggunakan Remote
int Idle_speed_remote(int buttonPushCounter){

    /*kiri vertikal naik = ditambah*/
    if (ch3 >= 1700){
      Serial.println("Kecepatan kapal: "); // Write to serial Monitor
      buttonPushCounter = buttonPushCounter += 5;
      Serial.println(buttonPushCounter); // Write to serial Monitor
      delay(200);
      if (buttonPushCounter >=2000){
        buttonPushCounter = 1488;
        }     
      }
    /*kiri vertikal ditengah*/
    if (ch3 >= 1200 && ch3 < 1700){
      Serial.println("Kecepatan kapal: "); // Write to serial Monitor
      buttonPushCounter = buttonPushCounter += 0;
      Serial.println(buttonPushCounter); // Write to serial Monitor
      }
      
    /*kiri vertikal turun = dikurang*/
    if (ch3 < 1200 ){
      Serial.println("Kecepatan kapal: "); // Write to serial Monitor
      buttonPushCounter = buttonPushCounter -= 5;
      Serial.println(buttonPushCounter); // Write to serial Monitor
      delay(200);
      if (buttonPushCounter < 1000){
        buttonPushCounter = 1488;
        }
      }
    return buttonPushCounter;
}

// 9.Fungsi display remote idle
int displayButtonSpeedCounterRem()
{
  buttonPushCounter = Idle_speed_remote(buttonPushCounter);
//  Counter = checkMode(Counter);
  lcd.clear();
  lcd.setCursor(0, 0); // Set cursor position on line 1 position 0
  lcd.print("Speed Remote : "); // Write letter
  lcd.setCursor(0, 1); // Set cursor position on line 2 position 0
  lcd.print(buttonPushCounter); // Write letter
  return buttonPushCounter;
}

//10. Fungsi DisplayRemote
void DisplayRemote(){
  setColor(0, 255, 0);  // Green
  remote();
  lcd.clear();
  lcd.setCursor(1, 0);
  lcd.print("R");
  Serial.println("mode remote"); // Write letter
}

//11. Fungsi display button naik dan ganti mode
void displayButtonSpeedCounter()
{
  buttonPushCounter = SpeedUp(buttonPushCounter);
  setColor(255, 0, 0);  // red
//  Counter = checkMode(Counter);
  ServoMove.write(90);
  lcd.clear();
  lcd.setCursor(0, 0); // Set cursor position on line 1 position 0
  lcd.print("Speed : "); // Write letter
  lcd.setCursor(0, 1); // Set cursor position on line 1 position 0
  lcd.print(buttonPushCounter);
}

//12. Fungsi display button naik dan ganti mode
int displayButtonCounter()
{
  Counter = checkMode(Counter);
  return Counter;
}

void loop() {
  displayButtonCounter();
  ch1 = pulseIn(CH1, HIGH, 30000); 
  ch3 = pulseIn(CH3, HIGH, 30000); 
  ch5 = pulseIn(CH5, HIGH, 30000); 
  ch6 = pulseIn(CH6, HIGH, 30000); 

//Mode Autonomous menggunakan Remote
  while (ch5 < 1400 && ch6 > 1400) 
  {
    setColor(0, 0, 255);  // blue
    autonomous();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("A");
    ch5 = pulseIn(CH5, HIGH, 30000); //Agar bisa pindah mode idle
  }
//Mode Masuk Remote
  while (ch5 > 1400 && ch6 < 1400) 
  {
    DisplayRemote();
  }

  /*Mode Idle Remote*/
    if (ch6 > 1400 && ch5 > 1400) 
  {
  /*Setpoint thruster speed */
    setColor(255, 50, 0);  // Green
    ServoMove.write(90);
    ThrusterSpeed1.writeMicroseconds(1488); // Make sure value of thruster horizontal is 1000
    ThrusterSpeed2.writeMicroseconds(1488); // Make sure value of thruster horizontal is 1000
    displayButtonSpeedCounterRem();
  }

/* 2. Masuk mode autonomous ketika Ganjil*/
  while (Counter % 2 != 0 && ch5<1400 && ch6<1400)
  {
    setColor(0, 0, 255);  // blue
    autonomous();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("A");
    Serial.println("A");
    displayButtonCounter();
    ch5 = pulseIn(CH5, HIGH, 30000); //Agar bisa pindah mode remote
}

  /* 3. Masuk mode idle Ketika Genap*/
if (Counter % 2 == 0 && ch5<1400 && ch6<1400)
  {
   Serial.print("Vessel Speed Remote : "); // Write letter
   Serial.println(buttonPushCounter); // Write letter
   ThrusterSpeed1.writeMicroseconds(1488); // Make sure value of thruster horizontal is 1000
   ThrusterSpeed2.writeMicroseconds(1488); // Make sure value of thruster horizontal is 1000
   ServoMove.write(90);
   ServoMove2.write(90);
   displayButtonSpeedCounter();
  }
}
