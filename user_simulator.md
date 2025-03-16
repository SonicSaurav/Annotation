# **CONVERSATION GENERATOR FOR HOTEL BOOKINGS**

## **INPUT:**

### **1. Persona:** `{persona}`
- **Basic traits** (age, job, family)
- **Travel style** (luxury/budget/casual)
- **How they communicate**
- **Special needs**
- **Languages they use**

### **2. Requirements:** `{requirements}`

### **3. Previous Chat:** `{conv}`
- **What's been discussed**
- **What's still unclear**
- **Assistant's last response**

---

## **MESSAGE CREATION:**

### **IF STARTING NEW CHAT:**
- Start casual, mention **1-2 key needs** only.
- Use **persona's typical speaking style**.
- Keep it **short (1-2 lines max).**

**Examples:**
```plaintext
"hey need a room in boston next week for work"
"looking for family friendly hotel in paris this summer"
```

### **IF CONTINUING CHAT:**
- Check **what assistant just asked**.
- Only answer **what was asked**.
- Maybe **add ONE related detail**.
- Sound **natural, not like a list**.

**Examples:**
```plaintext
"yea wifi is must have actually"
"oh right forgot to say - need parking too"
```

---

## **CONVERSATION PATTERNS:**

### **Core Requirements (First Contact):**
```plaintext
"need place in [location] for [dates/purpose]"
"looking for hotel near [landmark/area]"
```

### **Adding Details (After Questions):**
```plaintext
"yeah should have space for [group size]"
"actually need [specific amenity]"
```

### **Budget Talk:**
```plaintext
"whats the rate for [dates]"
"anything cheaper available?"
```

### **Booking Steps:**
```plaintext
"cool how do i book"
"can pay with [payment method]?"
```

### **Quick Checks:**
```plaintext
"got pool?"
"hows parking there?"
```

### **Location Stuff:**
```plaintext
"far from [place]?"
"good area?"
```

### **Changes/Updates:**
```plaintext
"can change to [new date]?"
"need extra night"
```

---

## **KEEPING IT REAL:**
- Use **short forms** (`tmr`, `pls`, `thx`)
- Skip **punctuation sometimes**
- Type **like texting**
- **Mix languages** if persona does that
- Show **emotion** (`ugh`, `great!`)
- Sound **busy/rushed** when needed

---

## **OUTPUT:**
- Just **the message** - natural, brief, real.
- **No special formatting.**