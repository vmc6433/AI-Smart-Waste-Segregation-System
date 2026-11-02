#include <Arduino.h>
#include <Servo.h>

// ============================================================
//  AI WASTE SEGREGATION — MOTION EXECUTOR  (protocol v6.6)
//  Fixed: STATE is sent BEFORE DONE, so Python always reads
//         fresh angle/position when a command completes.
// ============================================================

// ---------------- PIN MAP ----------------
const uint8_t DIR_PIN        = 2;
const uint8_t STEP_PIN       = 3;
const uint8_t MS1_PIN        = 4;
const uint8_t MS2_PIN        = 5;
const uint8_t MS3_PIN        = 6;
const uint8_t ENABLE_PIN     = 7;
const uint8_t SERVO_PIN      = 8;
const uint8_t SERVO_MAX_PIN  = 9;
const uint8_t SERVO_HOME_PIN = 10;
const uint8_t MOTOR_END_PIN  = 11;
const uint8_t MOTOR_HOME_PIN = 12;

// ---------------- SERVO RANGE ----------------
const int SERVO_HOME_ANGLE = 10;
const int SERVO_MAX_ANGLE  = 70;

// ---------------- TIMING ----------------
const uint16_t SERVO_TICK_MS      = 30;
const uint16_t SERVO_SETTLE_MS    = 250;
const uint16_t SERVO_VERIFY_MS    = 800;
const uint16_t SWITCH_CONFIRM_MS  = 25;

// Stepper
const uint16_t STEP_HIGH_US   = 5;
const uint16_t STEP_START_US  = 3200;
const uint16_t STEP_CRUISE_US = 1800;
const uint16_t STEP_ACCEL     = 10;
const uint16_t DIR_SETUP_US   = 20;
const long     MAX_STEPS      = 15000;

// ---------------- STATE ----------------
Servo tiltServo;
bool handshakeOK = false;
bool busy        = false;
int  servoAngle  = SERVO_HOME_ANGLE;
long motorPos    = 0;

char rxBuf[48];
uint8_t rxLen = 0;
volatile bool stopReq = false;

// ============================================================
//  SWITCH HELPERS
// ============================================================
inline bool shRaw() { return digitalRead(SERVO_HOME_PIN) == HIGH; }
inline bool smRaw() { return digitalRead(SERVO_MAX_PIN)  == HIGH; }
inline bool mhRaw() { return digitalRead(MOTOR_HOME_PIN) == HIGH; }
inline bool meRaw() { return digitalRead(MOTOR_END_PIN)  == HIGH; }

bool confirmSwitch(uint8_t pin, uint16_t ms = SWITCH_CONFIRM_MS) {
  uint8_t n = 4;
  uint16_t gap = ms / n; if (gap < 2) gap = 2;
  for (uint8_t i = 0; i < n; i++) {
    if (digitalRead(pin) != HIGH) return false;
    delay(gap);
  }
  return true;
}

bool waitForSwitch(uint8_t pin, uint16_t timeoutMs) {
  unsigned long start = millis();
  uint8_t good = 0;
  while (millis() - start < timeoutMs) {
    if (digitalRead(pin) == HIGH) {
      good++;
      if (good >= 3) return true;
    } else {
      good = 0;
    }
    delay(15);
  }
  return false;
}

void printSwitchStates() {
  Serial.print(F("DBG:SW,"));
  Serial.print(F("sH=")); Serial.print(shRaw() ? 1 : 0);
  Serial.print(F(",sM=")); Serial.print(smRaw() ? 1 : 0);
  Serial.print(F(",mH=")); Serial.print(mhRaw() ? 1 : 0);
  Serial.print(F(",mE=")); Serial.println(meRaw() ? 1 : 0);
}

// ============================================================
//  MOTOR PRIMITIVES
// ============================================================
inline void holdOn()  { digitalWrite(ENABLE_PIN, LOW);  }
inline void holdOff() { digitalWrite(ENABLE_PIN, HIGH); }

void setMicrosteps() {
  digitalWrite(MS1_PIN, HIGH);
  digitalWrite(MS2_PIN, HIGH);
  digitalWrite(MS3_PIN, HIGH);
}

void stepOnce(uint16_t intervalUs) {
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(STEP_HIGH_US);
  digitalWrite(STEP_PIN, LOW);
  if (intervalUs > STEP_HIGH_US) delayMicroseconds(intervalUs - STEP_HIGH_US);
}

uint16_t stepDelay(long i, long total) {
  if (total <= 4) return STEP_START_US;
  long edge = min(i, total - 1 - i);
  unsigned long red   = (unsigned long)edge * STEP_ACCEL;
  unsigned long avail = STEP_START_US - STEP_CRUISE_US;
  if (red > avail) red = avail;
  unsigned long v = STEP_START_US - red;
  if (v < STEP_CRUISE_US) v = STEP_CRUISE_US;
  return (uint16_t)v;
}

// ============================================================
//  STATUS
// ============================================================
void sendState() {
  Serial.print(F("STATE,"));
  Serial.print(servoAngle); Serial.print(',');
  Serial.print(motorPos); Serial.print(',');
  Serial.print(shRaw() ? 1 : 0); Serial.print(',');
  Serial.print(smRaw() ? 1 : 0); Serial.print(',');
  Serial.print(mhRaw() ? 1 : 0); Serial.print(',');
  Serial.print(meRaw() ? 1 : 0); Serial.print(',');
  Serial.print(1);
  Serial.print(',');
  Serial.println(busy ? 1 : 0);
}

void sendLimits() {
  Serial.print(F("LIMITS,"));
  Serial.print(shRaw() ? 1 : 0); Serial.print(',');
  Serial.print(smRaw() ? 1 : 0); Serial.print(',');
  Serial.print(mhRaw() ? 1 : 0); Serial.print(',');
  Serial.println(meRaw() ? 1 : 0);
}

// ============================================================
//  STOP PARSER
// ============================================================
bool pumpStop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      rxBuf[rxLen] = '\0';
      if (strcmp(rxBuf, "STOP") == 0) stopReq = true;
      rxLen = 0;
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    } else {
      rxLen = 0;
    }
  }
  return stopReq;
}

void sendStopped() {
  stopReq = false;
  busy = false;
  digitalWrite(STEP_PIN, LOW);
  holdOn();
  Serial.println(F("STOPPED"));
  sendState();
}

// ============================================================
//  SERVO PRIMITIVES
// ============================================================
void servoAttach() {
  if (tiltServo.attached()) return;
  int a = constrain(servoAngle, SERVO_HOME_ANGLE, SERVO_MAX_ANGLE);
  tiltServo.attach(SERVO_PIN);
  tiltServo.write(a);
  delay(SERVO_SETTLE_MS);
}

void servoDetach() {
  if (tiltServo.attached()) tiltServo.detach();
}

bool servoStepOnce(int dir) {
  int next = servoAngle + dir;
  if (next < SERVO_HOME_ANGLE) next = SERVO_HOME_ANGLE;
  if (next > SERVO_MAX_ANGLE)  next = SERVO_MAX_ANGLE;
  if (next == servoAngle) return false;
  servoAngle = next;
  tiltServo.write(servoAngle);
  delay(SERVO_TICK_MS);
  return true;
}

bool servoMoveTo(int target) {
  target = constrain(target, SERVO_HOME_ANGLE, SERVO_MAX_ANGLE);

  Serial.print(F("DBG:SM_START,target="));
  Serial.print(target);
  Serial.print(F(",from="));
  Serial.println(servoAngle);

  uint8_t shConfirm = 0;
  uint8_t smConfirm = 0;
  int steps = 0;

  while (servoAngle != target) {
    if (pumpStop()) {
      Serial.println(F("DBG:SM_STOP"));
      return false;
    }

    int dir = (target > servoAngle) ? +1 : -1;

    if (dir > 0) {
      if (smRaw()) {
        smConfirm++;
        if (smConfirm >= 2) {
          Serial.print(F("DBG:SM_MAXHIT,steps="));
          Serial.print(steps);
          Serial.print(F(",at="));
          Serial.println(servoAngle);
          // Do NOT push further — hold current position.
          servoAngle = SERVO_MAX_ANGLE;
          delay(SERVO_SETTLE_MS);
          return true;
        }
      } else {
        smConfirm = 0;
      }
    } else {
      if (shRaw()) {
        shConfirm++;
        if (shConfirm >= 2) {
          Serial.print(F("DBG:SM_HOMEHIT,steps="));
          Serial.print(steps);
          Serial.print(F(",at="));
          Serial.println(servoAngle);
          servoAngle = SERVO_HOME_ANGLE;
          tiltServo.write(SERVO_HOME_ANGLE);
          delay(SERVO_SETTLE_MS);
          return true;
        }
      } else {
        shConfirm = 0;
      }
    }

    if (!servoStepOnce(dir)) {
      Serial.print(F("DBG:SM_RANGE,at="));
      Serial.println(servoAngle);
      return true;
    }
    steps++;
    if (steps % 20 == 0) {
      Serial.print(F("DBG:SM_STEP,n="));
      Serial.print(steps);
      Serial.print(F(",ang="));
      Serial.println(servoAngle);
    }
  }

  Serial.print(F("DBG:SM_END,steps="));
  Serial.println(steps);
  return true;
}

// ============================================================
//  SERVO COMMANDS  (STATE is sent BEFORE DONE)
// ============================================================
bool cmdServoHome() {
  Serial.println(F("DBG:CH_START"));
  busy = true;
  servoAttach();
  Serial.println(F("START,SERVO_HOME"));

  if (confirmSwitch(SERVO_HOME_PIN, 40)) {
    Serial.println(F("DBG:CH_FASTPATH"));
    servoAngle = SERVO_HOME_ANGLE;
    tiltServo.write(SERVO_HOME_ANGLE);
    busy = false;
    sendState();
    Serial.println(F("DONE,SERVO_HOME"));
    return true;
  }

  bool atHome = false;
  for (uint8_t attempt = 1; attempt <= 3 && !atHome; attempt++) {
    Serial.print(F("DBG:CH_TRY,"));
    Serial.println(attempt);

    if (attempt > 1) {
      int back = constrain(servoAngle + 12, SERVO_HOME_ANGLE, SERVO_MAX_ANGLE);
      tiltServo.write(back);
      servoAngle = back;
      delay(320);
    }

    if (!servoMoveTo(SERVO_HOME_ANGLE)) {
      sendStopped();
      return false;
    }

    delay(SERVO_SETTLE_MS);
    atHome = waitForSwitch(SERVO_HOME_PIN, SERVO_VERIFY_MS);

    Serial.print(F("DBG:CH_VERIFY,"));
    Serial.println(atHome ? F("OK") : F("FAIL"));
    printSwitchStates();
  }

  servoAngle = SERVO_HOME_ANGLE;
  tiltServo.write(SERVO_HOME_ANGLE);
  delay(150);
  busy = false;

  if (!atHome) {
    Serial.println(F("ERROR,SERVO_HOME_NOT_REACHED"));
    sendState();
    return false;
  }

  sendState();
  Serial.println(F("DONE,SERVO_HOME"));
  return true;
}

bool cmdServoMax() {
  Serial.println(F("DBG:CM_START"));
  busy = true;
  servoAttach();
  Serial.println(F("START,SERVO_MAX"));

  if (smRaw()) {
    Serial.println(F("DBG:CM_FASTPATH"));
    servoAngle = SERVO_MAX_ANGLE;
    tiltServo.write(SERVO_MAX_ANGLE);
    busy = false;
    sendState();
    Serial.println(F("DONE,SERVO_MAX"));
    return true;
  }

  if (!servoMoveTo(SERVO_MAX_ANGLE)) {
    Serial.println(F("DBG:CM_SWEEP_ABORT"));
    sendStopped();
    return false;
  }

  delay(SERVO_SETTLE_MS);
  Serial.println(F("DBG:CM_VERIFY_BEGIN"));
  bool atMax = waitForSwitch(SERVO_MAX_PIN, SERVO_VERIFY_MS);

  Serial.print(F("DBG:CM_VERIFY,"));
  Serial.println(atMax ? F("OK") : F("FAIL"));
  printSwitchStates();

  busy = false;

  if (!atMax) {
    Serial.println(F("WARN,SERVO_MAX_SWITCH_NOT_PRESSED"));
  }
  sendState();
  Serial.println(F("DONE,SERVO_MAX"));
  return true;
}

// ============================================================
//  NEMA COMMANDS
// ============================================================
bool servoAtHome() { return confirmSwitch(SERVO_HOME_PIN, 40); }

bool cmdNemaMove(long steps) {
  if (steps == 0) {
    sendState();
    Serial.println(F("DONE,NEMA_MOVE,0"));
    return true;
  }
  if (!servoAtHome()) { Serial.println(F("ERROR,SERVO_NOT_HOME")); return false; }
  long total = labs(steps);
  if (total > MAX_STEPS) { Serial.println(F("ERROR,TOO_MANY_STEPS")); return false; }

  bool cw = (steps > 0);
  if (cw) {
    if (meRaw()) { Serial.println(F("ERROR,ALREADY_AT_END")); return false; }
    digitalWrite(DIR_PIN, HIGH); delayMicroseconds(DIR_SETUP_US);
  } else {
    if (mhRaw()) {
      motorPos = 0;
      sendState();
      Serial.println(F("DONE,NEMA_MOVE,0"));
      return true;
    }
    digitalWrite(DIR_PIN, LOW); delayMicroseconds(DIR_SETUP_US);
  }

  busy = true; holdOn();
  Serial.print(F("START,NEMA_MOVE,")); Serial.println(steps);

  unsigned long servoLostAt = 0;
  for (long i = 0; i < total; i++) {
    if (pumpStop()) { sendStopped(); return false; }

    if (!shRaw()) {
      if (!servoLostAt) servoLostAt = millis();
      if (millis() - servoLostAt >= 15) {
        busy = false;
        Serial.println(F("ERROR,SERVO_LEFT_HOME"));
        sendState(); return false;
      }
    } else servoLostAt = 0;

    if (cw && meRaw()) {
      busy = false; Serial.println(F("ERROR,END_LIMIT_HIT"));
      sendState(); return false;
    }
    if (!cw && mhRaw()) {
      motorPos = 0; busy = false;
      sendState();
      Serial.println(F("DONE,NEMA_MOVE_REACHED_HOME"));
      return true;
    }

    stepOnce(stepDelay(i, total));
    motorPos += cw ? 1 : -1;
  }

  busy = false;
  sendState();
  Serial.print(F("DONE,NEMA_MOVE,")); Serial.println(steps);
  return true;
}

bool cmdNemaHome() {
  if (!servoAtHome()) { Serial.println(F("ERROR,SERVO_NOT_HOME")); return false; }
  busy = true; holdOn();
  Serial.println(F("START,NEMA_HOME"));

  if (confirmSwitch(MOTOR_HOME_PIN, 40)) {
    motorPos = 0; busy = false;
    sendState();
    Serial.println(F("DONE,NEMA_HOME"));
    return true;
  }

  digitalWrite(DIR_PIN, LOW); delayMicroseconds(DIR_SETUP_US);
  unsigned long servoLostAt = 0;

  for (long i = 0; i < MAX_STEPS; i++) {
    if (pumpStop()) { sendStopped(); return false; }

    if (!shRaw()) {
      if (!servoLostAt) servoLostAt = millis();
      if (millis() - servoLostAt >= 15) {
        busy = false;
        Serial.println(F("ERROR,SERVO_LEFT_HOME"));
        sendState(); return false;
      }
    } else servoLostAt = 0;

    if (meRaw()) {
      busy = false;
      Serial.println(F("ERROR,END_LIMIT_HIT_DURING_HOME"));
      sendState(); return false;
    }

    if (confirmSwitch(MOTOR_HOME_PIN, 8)) {
      motorPos = 0; busy = false;
      sendState();
      Serial.println(F("DONE,NEMA_HOME"));
      return true;
    }

    stepOnce(stepDelay(i, MAX_STEPS));
    motorPos--;
  }

  busy = false;
  Serial.println(F("ERROR,NEMA_HOME_NOT_FOUND"));
  sendState();
  return false;
}

bool cmdNemaEnd() {
  if (!servoAtHome()) { Serial.println(F("ERROR,SERVO_NOT_HOME")); return false; }
  busy = true; holdOn();
  Serial.println(F("START,NEMA_END"));

  if (meRaw()) {
    busy = false;
    sendState();
    Serial.println(F("DONE,NEMA_END"));
    return true;
  }

  digitalWrite(DIR_PIN, HIGH); delayMicroseconds(DIR_SETUP_US);
  unsigned long servoLostAt = 0;

  for (long i = 0; i < MAX_STEPS; i++) {
    if (pumpStop()) { sendStopped(); return false; }

    if (!shRaw()) {
      if (!servoLostAt) servoLostAt = millis();
      if (millis() - servoLostAt >= 15) {
        busy = false;
        Serial.println(F("ERROR,SERVO_LEFT_HOME"));
        sendState(); return false;
      }
    } else servoLostAt = 0;

    if (mhRaw() && motorPos > 100) {
      busy = false;
      Serial.println(F("ERROR,HOME_LIMIT_HIT_DURING_END"));
      sendState(); return false;
    }

    if (confirmSwitch(MOTOR_END_PIN, 8)) {
      busy = false;
      sendState();
      Serial.println(F("DONE,NEMA_END"));
      return true;
    }

    stepOnce(stepDelay(i, MAX_STEPS));
    motorPos++;
  }

  busy = false;
  Serial.println(F("ERROR,NEMA_END_NOT_FOUND"));
  sendState();
  return false;
}

// ============================================================
//  CALIBRATION
// ============================================================
bool cmdNemaFindEnd() {
  if (!servoAtHome()) { Serial.println(F("ERROR,SERVO_NOT_HOME")); return false; }
  busy = true; holdOn();
  Serial.println(F("START,NEMA_FIND_END"));

  if (meRaw()) {
    busy = false;
    sendState();
    Serial.println(F("DONE,NEMA_FIND_END,0"));
    return true;
  }

  digitalWrite(DIR_PIN, HIGH); delayMicroseconds(DIR_SETUP_US);
  long count = 0;
  unsigned long servoLostAt = 0;

  for (long i = 0; i < MAX_STEPS; i++) {
    if (pumpStop()) { sendStopped(); return false; }
    if (!shRaw()) {
      if (!servoLostAt) servoLostAt = millis();
      if (millis() - servoLostAt >= 15) {
        busy = false;
        Serial.println(F("ERROR,SERVO_LEFT_HOME"));
        sendState(); return false;
      }
    } else servoLostAt = 0;

    if (meRaw()) {
      busy = false;
      sendState();
      Serial.print(F("DONE,NEMA_FIND_END,")); Serial.println(count);
      return true;
    }
    stepOnce(stepDelay(i, MAX_STEPS));
    motorPos++; count++;
  }

  busy = false;
  Serial.println(F("ERROR,NEMA_END_NOT_FOUND"));
  sendState();
  return false;
}

bool cmdNemaFindHome() {
  if (!servoAtHome()) { Serial.println(F("ERROR,SERVO_NOT_HOME")); return false; }
  busy = true; holdOn();
  Serial.println(F("START,NEMA_FIND_HOME"));

  if (confirmSwitch(MOTOR_HOME_PIN, 40)) {
    motorPos = 0; busy = false;
    sendState();
    Serial.println(F("DONE,NEMA_FIND_HOME,0"));
    return true;
  }

  digitalWrite(DIR_PIN, LOW); delayMicroseconds(DIR_SETUP_US);
  long count = 0;
  unsigned long servoLostAt = 0;

  for (long i = 0; i < MAX_STEPS; i++) {
    if (pumpStop()) { sendStopped(); return false; }
    if (!shRaw()) {
      if (!servoLostAt) servoLostAt = millis();
      if (millis() - servoLostAt >= 15) {
        busy = false;
        Serial.println(F("ERROR,SERVO_LEFT_HOME"));
        sendState(); return false;
      }
    } else servoLostAt = 0;

    if (confirmSwitch(MOTOR_HOME_PIN, 8)) {
      motorPos = 0; busy = false;
      sendState();
      Serial.print(F("DONE,NEMA_FIND_HOME,")); Serial.println(count);
      return true;
    }
    stepOnce(stepDelay(i, MAX_STEPS));
    motorPos--; count++;
  }

  busy = false;
  Serial.println(F("ERROR,NEMA_HOME_NOT_FOUND"));
  sendState();
  return false;
}

// ============================================================
//  COMMAND DISPATCH
// ============================================================
void processCommand(const char *cmd) {
  if (strcmp(cmd, "HELLO") == 0) {
    handshakeOK = true;
    holdOn();
    Serial.println(F("HELLO_ACK,WASTE_MOTION,6"));
    sendState();
    return;
  }
  if (strcmp(cmd, "STATUS") == 0) {
    if (!handshakeOK) { Serial.println(F("ERROR,NO_HANDSHAKE")); return; }
    sendState(); return;
  }
  if (strcmp(cmd, "LIMITS") == 0) {
    if (!handshakeOK) { Serial.println(F("ERROR,NO_HANDSHAKE")); return; }
    sendLimits(); return;
  }
  if (strcmp(cmd, "STOP") == 0) {
    if (busy) stopReq = true;
    else      sendStopped();
    return;
  }
  if (!handshakeOK) { Serial.println(F("ERROR,NO_HANDSHAKE")); return; }
  if (busy)         { Serial.println(F("ERROR,BUSY"));          return; }

  if (strcmp(cmd, "SERVO_HOME") == 0) { cmdServoHome(); return; }
  if (strcmp(cmd, "SERVO_MAX")  == 0) { cmdServoMax();  return; }

  if (strncmp(cmd, "NEMA_CW ",  8)  == 0) { cmdNemaMove(atol(cmd + 8));   return; }
  if (strncmp(cmd, "NEMA_CCW ", 9)  == 0) { cmdNemaMove(-atol(cmd + 9));  return; }

  if (strcmp(cmd, "NEMA_HOME")      == 0) { cmdNemaHome();     return; }
  if (strcmp(cmd, "NEMA_END")       == 0) { cmdNemaEnd();      return; }
  if (strcmp(cmd, "NEMA_FIND_END")  == 0) { cmdNemaFindEnd();  return; }
  if (strcmp(cmd, "NEMA_FIND_HOME") == 0) { cmdNemaFindHome(); return; }

  Serial.println(F("ERROR,UNKNOWN_COMMAND"));
}

void readSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      rxBuf[rxLen] = '\0';
      if (rxLen > 0) processCommand(rxBuf);
      rxLen = 0;
      continue;
    }
    if (rxLen < sizeof(rxBuf) - 1) rxBuf[rxLen++] = c;
    else { rxLen = 0; Serial.println(F("ERROR,CMD_TOO_LONG")); }
  }
}

void setup() {
  pinMode(DIR_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(MS1_PIN, OUTPUT);
  pinMode(MS2_PIN, OUTPUT);
  pinMode(MS3_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  pinMode(SERVO_MAX_PIN,  INPUT_PULLUP);
  pinMode(SERVO_HOME_PIN, INPUT_PULLUP);
  pinMode(MOTOR_END_PIN,  INPUT_PULLUP);
  pinMode(MOTOR_HOME_PIN, INPUT_PULLUP);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);
  setMicrosteps();
  holdOn();
  servoDetach();

  Serial.begin(115200);
  delay(1000);
  Serial.println(F("UNO_READY,WASTE_MOTION,6"));
  sendState();
}

void loop() { readSerial(); }