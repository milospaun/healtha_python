#CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(190) NOT NULL,
    lastname VARCHAR(190) NOT NULL,
    email VARCHAR(190) UNIQUE NOT NULL,
    password VARCHAR(190) NOT NULL,
    role VARCHAR(190),
    role_id INTEGER,
    created_at TIMESTAMP
)

#CREATE TABLE IF NOT EXISTS data(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    gx DOUBLE(12,6),
    gy DOUBLE(12,6),
    gz DOUBLE(12,6),
    ax DOUBLE(12,6),
    ay DOUBLE(12,6),
    az DOUBLE(12,6),
    timestamp TIMESTAMP,
    user_id INTEGER,
    source ENUM('ios', 'android', 'watch'),
    created_at TIMESTAMP,
    processed BOOLEAN DEFAULT false,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)

#CREATE TABLE IF NOT EXISTS statistics(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    total INTEGER DEFAULT 0,
    walking INTEGER DEFAULT 0,
    walking_upstairs INTEGER DEFAULT 0,
    walking_downstairs INTEGER DEFAULT 0,
    sitting INTEGER DEFAULT 0,
    standing INTEGER DEFAULT 0,
    laying INTEGER DEFAULT 0,
    stand_to_sit INTEGER DEFAULT 0,
    sit_to_stand INTEGER DEFAULT 0,
    sit_to_lie INTEGER DEFAULT 0,
    lie_to_sit INTEGER DEFAULT 0,
    stand_to_lie INTEGER DEFAULT 0,
    lie_to_stand INTEGER DEFAULT 0,
    user_id INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)