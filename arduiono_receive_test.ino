#include <AFMotor.h>
#include <Servo.h>

AF_DCMotor motor(1,MOTOR12_2KHZ);
Servo myservo;

void setup() {
  // put your setup code here, to run once:

  pinMode(LED_BUILTIN, OUTPUT);

  motor.setSpeed(200);
  
  myservo.attach(9);

  
  Serial.begin(9600);
  while (!Serial) {
    ;
  }

}

void loop() {
  // put your main code here, to run repeatedly:
  char buffer[16];

  
  
  if (Serial.available() > 0) {
    int size = Serial.readBytesUntil('\n', buffer, 12);
    
    if (buffer[0] == 'Y') {
      digitalWrite(LED_BUILTIN, HIGH);
      motor.run(FORWARD);
      if(myservo.read() == 0){
        myservo.write(180);
      }else{
        myservo.write(0);
      }
    }
    if (buffer[0] == 'N') {
      digitalWrite(LED_BUILTIN, HIGH);
      motor.run(RELEASE);
      myservo.write(90);
    }
  }

}
