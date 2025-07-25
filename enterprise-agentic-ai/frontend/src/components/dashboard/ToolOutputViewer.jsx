import React from 'react';

const ToolOutputViewer = ({ output }) => {
    return (
        <div className="tool-output-viewer">
            <h2>Tool Output</h2>
            <pre>{output}</pre>
        </div>
    );
};

export default ToolOutputViewer;