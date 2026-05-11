const int outputPin = 9;

String input = "";

void setup() {
  pinMode(outputPin, OUTPUT);
  digitalWrite(outputPin, LOW);

  Serial.begin(115200);

  while (!Serial) {
    ;
  }

  Serial.println("Ready");
}

void loop() {

  while (Serial.available()) {

    char c = Serial.read();

    if (c == '\n') {

      input.trim();

      if (input == "HIGH") {
        digitalWrite(outputPin, HIGH);
      }

      else if (input == "LOW") {
        digitalWrite(outputPin, LOW);
      }

      input = "";
    }

    else {
      input += c;
    }
  }
}