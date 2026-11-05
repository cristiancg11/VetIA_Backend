-- ==============================================================================
-- VETIA / PETEMERGENCY - SUPABASE SQL SCHEMA, RLS POLICIES & PROFILE SYNC
-- ==============================================================================
-- Ejecuta este script en Supabase Dashboard > SQL Editor para dejar la base de datos
-- y el almacenamiento 100% operativos. Resuelve los errores de políticas RLS (42501).
-- ==============================================================================

-- 1. Asegurar la extensión para UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Tabla de Registros de Triage (triage_records)
CREATE TABLE IF NOT EXISTS public.triage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    pet_type TEXT NOT NULL,
    symptoms_description TEXT NOT NULL,
    image_url TEXT NOT NULL,
    urgency_level TEXT NOT NULL,
    urgency_color TEXT NOT NULL,
    preliminary_assessment TEXT NOT NULL,
    immediate_care_tips JSONB NOT NULL DEFAULT '[]'::jsonb,
    recommended_facility_type TEXT NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    user_lat DOUBLE PRECISION,
    user_lng DOUBLE PRECISION
);

-- Habilitar Row Level Security (RLS) en triage_records
ALTER TABLE public.triage_records ENABLE ROW LEVEL SECURITY;

-- Políticas de RLS para triage_records:
-- Permite lectura pública o por usuario
DROP POLICY IF EXISTS "Permitir lectura de triage_records" ON public.triage_records;
CREATE POLICY "Permitir lectura de triage_records"
    ON public.triage_records FOR SELECT
    USING (true);

-- Permite inserción a usuarios autenticados y anónimos (frontend / backend con anon key)
DROP POLICY IF EXISTS "Permitir insercion de triage_records" ON public.triage_records;
CREATE POLICY "Permitir insercion de triage_records"
    ON public.triage_records FOR INSERT
    WITH CHECK (true);

-- Permite actualizar o eliminar solo los registros propios si tienen user_id
DROP POLICY IF EXISTS "Permitir actualizacion de sus propios triage_records" ON public.triage_records;
CREATE POLICY "Permitir actualizacion de sus propios triage_records"
    ON public.triage_records FOR UPDATE
    USING (auth.uid() = user_id);

-- Índice para acelerar consultas de historial filtrado por tutor
CREATE INDEX IF NOT EXISTS idx_triage_records_user_id ON public.triage_records(user_id);
CREATE INDEX IF NOT EXISTS idx_triage_records_created_at ON public.triage_records(created_at DESC);


-- 3. Tabla de Perfiles en el esquema público (profiles)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Habilitar Row Level Security (RLS) en profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Políticas de acceso para profiles
DROP POLICY IF EXISTS "Permitir lectura de perfiles" ON public.profiles;
CREATE POLICY "Permitir lectura de perfiles"
    ON public.profiles FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Permitir a los usuarios insertar o editar su propio perfil" ON public.profiles;
CREATE POLICY "Permitir a los usuarios insertar o editar su propio perfil"
    ON public.profiles FOR ALL
    USING (auth.uid() = id);


-- 4. Función y Trigger automático para sincronizar auth.users -> public.profiles
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, created_at, updated_at)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1)),
        NOW(),
        NOW()
    )
    ON CONFLICT (id) DO UPDATE
    SET
        email = EXCLUDED.email,
        full_name = EXCLUDED.full_name,
        updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Migrar e insertar usuarios ya existentes en auth.users hacia public.profiles
INSERT INTO public.profiles (id, email, full_name, created_at, updated_at)
SELECT 
    id, 
    email, 
    COALESCE(raw_user_meta_data->>'full_name', split_part(email, '@', 1)),
    created_at,
    NOW()
FROM auth.users
ON CONFLICT (id) DO NOTHING;


-- 5. Configuración del Bucket de Almacenamiento (pet-images) en Storage
INSERT INTO storage.buckets (id, name, public)
VALUES ('pet-images', 'pet-images', true)
ON CONFLICT (id) DO UPDATE SET public = true;

-- Políticas de Storage para el bucket pet-images
DROP POLICY IF EXISTS "Permitir lectura publica de imagenes de mascotas" ON storage.objects;
CREATE POLICY "Permitir lectura publica de imagenes de mascotas"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'pet-images');

DROP POLICY IF EXISTS "Permitir subida publica de imagenes de mascotas" ON storage.objects;
CREATE POLICY "Permitir subida publica de imagenes de mascotas"
    ON storage.objects FOR INSERT
    WITH CHECK (bucket_id = 'pet-images');

DROP POLICY IF EXISTS "Permitir actualizacion de imagenes de mascotas" ON storage.objects;
CREATE POLICY "Permitir actualizacion de imagenes de mascotas"
    ON storage.objects FOR UPDATE
    USING (bucket_id = 'pet-images');
