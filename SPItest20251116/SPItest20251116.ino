//Code for testing SPI operation
//2025.07.03
//Alexander Fahey
//Ported for standard Arduino Nano

#include <SPI.h>

char hexString[5];
short value;
int i;
short CycleLowValue;
short CycleHighValue;
int CycleSpeedValue;

const int CS_PIN = 10;  // D10 for reset

//The following code is meant to allow some simple tests of the SPI that can be run in succession without reuploading the code with minor changes.
//Basic Operation
//Give user 3 choices that user can select between:
//1) Send the reset signal and automatically return to the menu.
//2) Allow user to send one command in hex format and automatically return to the menu.
//3) Allow user to cycle through set of possible commands and set begin loop, end loop, and cycle speed
  //Also, show end command while cycling to allow returning to menu.

void setup() {
  Serial.begin(115200);     // Initialization of serial communication.
  
  // Initialize CS pin
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, LOW);  // Start LOW (idle state)
  
  // Initialize SPI
  SPI.begin();
  SPI.setClockDivider(SPI_CLOCK_DIV2);  // ~8MHz on 16MHz Nano
  SPI.setDataMode(SPI_MODE0);  // SPI Mode 0 (clock idles LOW)
  SPI.setBitOrder(MSBFIRST);
  
  MyReset();
}

void MyReset() {
  //Sends a 1ms high pulse on the Chip Select. This is connected to SRST to serve as an Active-High external reset.
  Serial.println("SPI Reset");
  digitalWrite(CS_PIN, HIGH);  // Drive the CS LOW.
  delay(1);
  digitalWrite(CS_PIN, LOW);  // Drive the CS HIGH.
  delay(1);
}

void MyCommand() {
  //Waits for 4 chars the user should send in due to early prompt and then puts them in an array.
  //This new array of 4 chars is translated into a hex value and written to the SPI.
    //Chars are assumed to be ones with Hexadecimal equivalents (e.g., 0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F)
  while (true) {
    Serial.println(Serial.available());
    if (Serial.available() > 4) {
      break;
    }
    delay(5000);
  }
  i = 0;
  while (i < 4) {
    hexString[i] = Serial.read();
    Serial.println(hexString[i]);
    i += 1;
  }
  hexString[4] = '\0';  // Null terminate
  value = strtol(hexString, NULL, 16);
  Serial.println(value);
  SPI.transfer16(value);
}

void SetCycleLowValue() {
  //Waits for 4 chars the user should send in due to early prompt and then puts them in an array.
  //This new array of 4 chars is translated into a hex value and saved for the cycling command.
    //Chars are assumed to be ones with Hexadecimal equivalents (e.g., 0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F)
  while (true) {
    if (Serial.available() > 4) {
      break;
    }
    delay(5000);
  }
  i = 0;
  while (i < 4) {
    hexString[i] = Serial.read();
    Serial.println(hexString[i]);
    i += 1;
  }
  hexString[4] = '\0';  // Null terminate
  CycleLowValue = strtol(hexString, NULL, 16);
  Serial.readString();
}

void SetCycleHighValue() {
  //Waits for 4 chars the user should send in due to early prompt and then puts them in an array.
  //This new array of 4 chars is translated into a hex value and saved for the cycling command.
    //Chars are assumed to be ones with Hexadecimal equivalents (e.g., 0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F)
  while (true) {
    if (Serial.available() > 4) {
      break;
    }
    delay(5000);
  }
  i = 0;
  while (i < 4) {
    hexString[i] = Serial.read();
    Serial.println(hexString[i]);
    i += 1;
  }
  hexString[4] = '\0';  // Null terminate
  CycleHighValue = strtol(hexString, NULL, 16);
  Serial.readString();
}

void SetCycleSpeedValue() {
  //Waits for chars the user should send in due to early prompt and then converts them into a decimal value and saves it to a variable.
    //Chars are assumed to be ones with decimal equivalents (e.g., 0,1,2,3,4,5,6,7,8,9)
    //-----Also, apparent speed is clearly different from this value. Not sure what I missed, but I will likely find the issue and fix it later.
  while (true) {
    if (Serial.available() > 1) {
      break;
    }
    delay(5000);
  }
  String cmdstr = Serial.readString();
  Serial.println(cmdstr);
  CycleSpeedValue = cmdstr.toInt();
  Serial.println(CycleSpeedValue);
}

void MyCyclingCommand() {
  //Uses accumulated data to write every value between the low and high value to the SPI with the speed modulating 
  //how quickly it does s0. This process will loop back to the low value once it reaches the high value.
  value = CycleLowValue;
  while (true) {

    //Serial.println("Enter '0' to exit Loop.");       //if you uncomment this block you will be able to exit the loop without reseting but it will be much slower
    //String cmd = Serial.readStringUntil('\n');
    //if (String(cmd) == "0") {
    //  break;
    //}
    
    if (value == CycleHighValue) {
      value = CycleLowValue;
    }
    Serial.println(value);
    SPI.transfer16(value);
    value += 1;
    Serial.println(CycleSpeedValue);
    delay(50);
    //receivedData = SPI.transfer(dataToSend);
  }
}

void loop() {
  Serial.println();
  Serial.println("Enter '0' to send an external reset.");
  Serial.println("Enter '1' to send a hexadecimal command.");
  Serial.println("Enter '2' to set up cycling through commands.");
  Serial.println("Press Enter to finish command.");
  
  // Wait for user input before proceeding
  while (Serial.available() == 0) {
    // Block here until user sends something
  }
  String cmd = Serial.readStringUntil('\n');
  
  if (String(cmd) == "0") {
    MyReset();
  } else if (String(cmd) == "1") {
    Serial.println("You can now send an SPI command.");
    Serial.println("Please enter a hexadecimal value that is 4 digits long (e.g., '4FA1').");
    MyCommand();
  } else if (String(cmd) == "2") {
    Serial.println("This process will ask you to input a low hexadecimal value, a high hexadecimal value,");
    Serial.println("and the speed to cycle through all commands between them, inclusive.");
    Serial.println("You can now write the low value you wish to use.");
    Serial.println("Please enter a hexadecimal value that is 4 digits long (e.g., '4FA1').");
    SetCycleLowValue();
    Serial.println("You can now write the high value you wish to use.");
    Serial.println("Please enter a hexadecimal value that is 4 digits long (e.g., '4FA1').");
    SetCycleHighValue();
    Serial.println("You can now write the desired cycle speed in miliseconds.");
    Serial.println("Please enter a whole number of any size.");
    SetCycleSpeedValue();
    Serial.println("Start Cycling---");
    MyCyclingCommand();
  }
  
  // No delay here - loop waits for next command input
}