import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const getEngagements = async () => {
    const response = await apiClient.get('/engagements');
    return response.data;
};

export const createEngagement = async (engagementData) => {
    const response = await apiClient.post('/engagements', engagementData);
    return response.data;
};

export const getAIModels = async () => {
    const response = await apiClient.get('/ai_agents');
    return response.data;
};

export const executeCommand = async (commandData) => {
    const response = await apiClient.post('/execution', commandData);
    return response.data;
};