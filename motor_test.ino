// Motor 1 connections (right motor)
int enA = 11;
int in1 = 13;
int in2 = 12;
// Motor 2 connections (left motor)
int enB = 6; ///9
int in3 = 4; //8
int in4 = 2; //7
// Motor 3 connections (up motor)
int enC = 6;
int in5 = 4;
int in6 = 2;

int val;
void setup() {
  

  //serioal
  Serial.begin(9600);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Set all the motor control pins to outputs
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(enC, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(in5, OUTPUT);
  pinMode(in6, OUTPUT);
  
  // Turn off motors - Initial state
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  digitalWrite(in5, LOW); 
  digitalWrite(in5, LOW);
}

void loop() {
  while(!Serial.available());
  val = Serial.readString().toInt();
  Serial.write(val);
  
  if (val == 0){
    turnLeft();
  } else if (val == 1){
    turnRight();
  } else if (val == 2){
    goUp();
  } else if(val == 3){
    goDown();
  } else if(val == 4){
    goForward();
  }
  //turnLeft();
 // delay(1000);
  //goUp();
  //delay(1000);
  //turnRight();
  //delay(1000);
 // goDown();
  //delay(1000);
  //goForward();
  //delay(1000);
}
void turnLeft(){
  analogWrite(enA, 110); //right motor
  analogWrite(enB, 120); //left motor
  
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  delay(4700);

  // Turn off motors
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

void turnRight(){
  analogWrite(enA, 130); //right motor
  analogWrite(enB, 120); //left motor
  
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(4700);

  // Turn off motors
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  
}

void goUp(){
  analogWrite(enC, 255);
  digitalWrite(in5, HIGH);
  digitalWrite(in6, LOW);
  delay(1000);
  digitalWrite(in5, LOW);
  digitalWrite(in6, LOW);
  
}

void goDown(){
  analogWrite(enC, 255);
  digitalWrite(in5, LOW);
  digitalWrite(in6, HIGH);
  delay(1000);
  digitalWrite(in5, LOW);
  digitalWrite(in6, LOW);
  
}
void goForward(){
  analogWrite(enA, 200); //right motor
  analogWrite(enB, 255); //left motor
  
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  delay(8000);

  // Turn off motors
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  
}
