-- ==============================================================================
-- VETIA / PETEMERGENCY - SUPABASE SQL SCHEMA & USER PROFILE SYNC MIGRATION
-- ==============================================================================
-- Ejecuta este script en Supabase Dashboard > SQL Editor para:
-- 1. Crear la tabla pública 'profiles' visible en el 'Table Editor'.
-- 2. Sincronizar automáticamente cualquier usuario nuevo de 'auth.users' a 'public.profiles'.
-- 3. Agregar la columna 'user_id' a 'triage_records' para vincular historiales por tutor.
-- ==============================================================================

-- 1. Tabla de Perfiles en el esquema público (visible en Table Editor)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Habilitar Row Level Security (RLS) en la tabla profiles
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

-- 2. Función y Trigger automático para sincronizar auth.users -> public.profiles
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

-- 3. Migrar e insertar usuarios ya existentes en auth.users hacia public.profiles
INSERT INTO public.profiles (id, email, full_name, created_at, updated_at)
SELECT 
    id, 
    email, 
    COALESCE(raw_user_meta_data->>'full_name', split_part(email, '@', 1)),
    created_at,
    NOW()
FROM auth.users
ON CONFLICT (id) DO NOTHING;

-- 4. Vincular columna user_id en la tabla triage_records
ALTER TABLE public.triage_records 
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- 5. Crear índice para acelerar consultas de historial filtrado por tutor
CREATE INDEX IF NOT EXISTS idx_triage_records_user_id ON public.triage_records(user_id);
