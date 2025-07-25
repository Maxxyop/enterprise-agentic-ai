import React from 'react';

const HindiReport = ({ reportData }) => {
    return (
        <div>
            <h1>रिपोर्ट</h1>
            <h2>सारांश</h2>
            <p>{reportData.summary}</p>
            <h2>मुख्य निष्कर्ष</h2>
            <ul>
                {reportData.findings.map((finding, index) => (
                    <li key={index}>{finding}</li>
                ))}
            </ul>
            <h2>सिफारिशें</h2>
            <p>{reportData.recommendations}</p>
        </div>
    );
};

export default HindiReport;